"""
Hardened sandbox for executing model-generated Python against unit tests.

This implements the canonical HumanEval pass@1 protocol: assemble a full program
(prompt prefix + candidate completion + test harness), execute it in an ISOLATED
subprocess, and report pass/fail. Modeled closely on OpenAI human-eval's
`execution.py:check_correctness` and its `reliability_guard`.

Isolation mechanics:
  - Subprocess per call. Start method = `fork` on Linux, `spawn` on macOS
    (env override `RLVR_SANDBOX_START`). This is the crux: under `spawn` the
    child RE-IMPORTS the `__main__` module — which in a training run is the
    trainer script that imports torch/transformers (~10s). That blows the join
    deadline and every program comes back as a false "timeout (no result)" —
    i.e. all rewards collapse to 0. `fork` makes the child a memory copy that
    never re-imports `__main__`, so candidate code runs immediately. This is
    what OpenAI human-eval does.
  - Hard wall-clock timeout: parent joins with a deadline, then terminates/kills,
    plus an in-child `signal.SIGALRM` backstop. This is the primary hang guard.
  - In-child `reliability_guard()`: disables filesystem-mutating + process calls
    (os.remove/rmdir/rename, shutil.rmtree, subprocess, write-mode helpers),
    clears the environment, and silences stdio.
  - RLIMIT_AS is OFF by default: under `fork` the child inherits the parent's
    (torch/CUDA) virtual address space, which is tens of GB, so capping AS would
    break inherited mappings. Memory is bounded by the OS sandbox (Slurm cgroup /
    Colab) instead. Pass `memory_limit_bytes` explicitly to opt back in.

SECURITY NOTE: this neutralizes accidental and casual-malicious code, but it is
NOT a true security boundary — a determined adversary can escape an in-process
guard. For untrusted code at scale use OS-level isolation (containers, gVisor,
firejail, seccomp). In this project the executor runs inside the Slurm job
sandbox on the cluster, which provides the real boundary.

Pure stdlib: this module must not import torch / datasets / transformers.
"""

from __future__ import annotations

import ast
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
from typing import Any, Dict, List, Optional, Tuple

# Default per-program wall-clock budget (seconds).
DEFAULT_TIMEOUT = 5.0
# Address-space cap is OFF by default — see the module docstring (fork inherits
# torch/CUDA virtual memory). Callers can opt in by passing memory_limit_bytes.
DEFAULT_MEMORY_LIMIT_BYTES = None


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float):
    """SIGALRM-based time limit (child process only)."""

    def _handler(signum, frame):
        raise TimeoutException("timed out")

    # setitimer supports sub-second resolution; SIGALRM only fires on the main
    # thread of the (child) process, which is where we exec the program.
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    """Redirect stdin/stdout/stderr to a null stream so candidate prints/inputs
    can't pollute the parent or block on input."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        with _redirect_stdin(stream):
            yield


@contextlib.contextmanager
def _redirect_stdin(new_target):
    old = __import__("sys").stdin
    import sys as _sys

    _sys.stdin = new_target
    try:
        yield
    finally:
        _sys.stdin = old


def reliability_guard(memory_limit_bytes: Optional[int] = DEFAULT_MEMORY_LIMIT_BYTES):
    """Disable functions that can mutate the host or spawn processes.

    Ported and trimmed from OpenAI human-eval's `reliability_guard`. Runs INSIDE
    the child process before executing the candidate program. This does not make
    the child safe against a determined attacker — it raises the cost of casual
    filesystem/network/process side effects and prevents hangs.
    """
    if memory_limit_bytes is not None:
        try:
            import resource

            # RLIMIT_AS is unreliable / breaks the interpreter on macOS.
            if platform.uname().system != "Darwin":
                resource.setrlimit(
                    resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes)
                )
                resource.setrlimit(
                    resource.RLIMIT_DATA, (memory_limit_bytes, memory_limit_bytes)
                )
        except Exception:
            pass

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None
    builtins.help = None

    # Block writes via the environment.
    os.environ["OMP_NUM_THREADS"] = "1"

    # Filesystem-mutating os calls.
    for _name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, _name):
            setattr(os, _name, None)

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    # NOTE: we intentionally do NOT disable `socket` here. The result is shipped
    # back from this child via a multiprocessing.Manager proxy, which uses a
    # socket; blocking sockets would break our own IPC. Network egress is not
    # part of the HumanEval/MBPP test protocol, and the real isolation boundary
    # is the OS-level sandbox (Slurm job / container) — see module docstring.

    import sys as _sys

    _sys.modules["ipdb"] = None
    _sys.modules["joblib"] = None
    _sys.modules["psutil"] = None
    _sys.modules["tkinter"] = None


def _unsafe_execute(
    program: str,
    timeout: float,
    memory_limit_bytes: Optional[int],
    result: List[str],
) -> None:
    """Child-process target. Appends a single outcome token to `result`."""
    reliability_guard(memory_limit_bytes)
    try:
        exec_globals: Dict[str, Any] = {}
        with _swallow_io():
            with _time_limit(timeout):
                # The assembled program runs the tests itself (e.g. ends in
                # `check(entry_point)` for HumanEval, or trailing asserts for MBPP).
                exec(program, exec_globals)
        result.append("passed")
    except TimeoutException:
        result.append("timeout")
    except AssertionError:
        result.append("failed")
    except BaseException as exc:  # noqa: BLE001 — capture everything from candidate
        result.append(f"exec_error: {type(exc).__name__}: {exc}")


def _select_start_method() -> str:
    """Pick a multiprocessing start method for sandbox children.

    `fork` (Linux default) is required when the executor is called from a
    torch/transformers training script: it does NOT re-import `__main__`, so the
    child doesn't reload torch and time out. `spawn` re-imports `__main__` and is
    only safe when that module is lightweight (e.g. macOS dev / standalone tests).
    Override with RLVR_SANDBOX_START={fork,spawn,forkserver}.
    """
    override = os.environ.get("RLVR_SANDBOX_START")
    available = multiprocessing.get_all_start_methods()
    if override and override in available:
        return override
    if platform.uname().system == "Darwin":
        return "spawn"  # fork is unsafe on macOS; __main__ is light in dev/tests
    if "fork" in available:
        return "fork"
    return "spawn"


_MP_CONTEXT = multiprocessing.get_context(_select_start_method())


def check_correctness(
    program: str,
    entry_point: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    memory_limit_bytes: Optional[int] = DEFAULT_MEMORY_LIMIT_BYTES,
) -> Dict[str, Any]:
    """Execute `program` in an isolated subprocess and report the outcome.

    Args:
        program: Full, self-running Python source (includes the test harness).
        entry_point: Expected function name; used only for a fast pre-check.
        timeout: Wall-clock budget in seconds.
        memory_limit_bytes: RLIMIT_AS/DATA cap inside the child (None to skip).

    Returns:
        {"passed": bool, "diagnostic": "passed"|"failed"|"timeout"|"syntax_error"|"exec_error",
         "detail": str}
    """
    # Fast, deterministic syntax pre-check — avoids spawning a process for code
    # that can't compile, and gives a clean "syntax_error" diagnostic.
    try:
        compile(program, "<candidate>", "exec")
    except SyntaxError as exc:
        return {"passed": False, "diagnostic": "syntax_error", "detail": str(exc)}

    manager = _MP_CONTEXT.Manager()
    result: List[str] = manager.list()  # type: ignore[assignment]

    proc = _MP_CONTEXT.Process(
        target=_unsafe_execute,
        args=(program, timeout, memory_limit_bytes, result),
    )
    proc.start()
    # Give the child a small grace period beyond its own SIGALRM budget so the
    # in-child timeout fires first when possible (cleaner diagnostic).
    proc.join(timeout + 1.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(0.5)
        if proc.is_alive():
            proc.kill()
            proc.join()

    if not result:
        return {"passed": False, "diagnostic": "timeout", "detail": "no result (killed)"}

    token = result[0]
    if token == "passed":
        return {"passed": True, "diagnostic": "passed", "detail": ""}
    if token == "timeout":
        return {"passed": False, "diagnostic": "timeout", "detail": "time limit exceeded"}
    if token == "failed":
        return {"passed": False, "diagnostic": "failed", "detail": "assertion failed"}
    # exec_error: <...>
    return {"passed": False, "diagnostic": "exec_error", "detail": token}


def check_correctness_batch(
    items: List[Tuple[str, Optional[str]]],
    timeout: float = DEFAULT_TIMEOUT,
    memory_limit_bytes: Optional[int] = DEFAULT_MEMORY_LIMIT_BYTES,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Score many (program, entry_point) pairs concurrently.

    Uses a spawn-context process pool so process-startup cost amortizes across a
    batch (RL generates B*G completions per step). Each task still executes in a
    fresh interpreter with `reliability_guard` applied. A task that exceeds the
    per-future deadline is reported as a timeout and its worker is discarded;
    the pool replaces it. Falls back to sequential `check_correctness` when a
    pool can't be created.

    Returns results in the same order as `items`.
    """
    if not items:
        return []

    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=_MP_CONTEXT)
    except Exception:
        return [
            check_correctness(prog, ep, timeout, memory_limit_bytes)
            for prog, ep in items
        ]

    try:
        future_to_idx = {}
        for idx, (program, entry_point) in enumerate(items):
            # Pre-check syntax in-parent (cheap) so syntactically broken programs
            # never occupy a worker.
            try:
                compile(program, "<candidate>", "exec")
            except SyntaxError as exc:
                results[idx] = {
                    "passed": False, "diagnostic": "syntax_error", "detail": str(exc),
                }
                continue
            fut = executor.submit(
                _pool_worker, program, timeout, memory_limit_bytes
            )
            future_to_idx[fut] = idx

        for fut, idx in future_to_idx.items():
            try:
                results[idx] = fut.result(timeout=timeout + 5.0)
            except FutureTimeout:
                results[idx] = {
                    "passed": False, "diagnostic": "timeout",
                    "detail": "worker exceeded deadline",
                }
            except Exception as exc:  # worker crashed
                results[idx] = {
                    "passed": False, "diagnostic": "exec_error",
                    "detail": f"worker error: {exc}",
                }
    finally:
        # Don't wait on stuck workers — let the OS reap them.
        executor.shutdown(wait=False, cancel_futures=True)

    return [r if r is not None else {
        "passed": False, "diagnostic": "exec_error", "detail": "no result",
    } for r in results]


def _pool_worker(
    program: str, timeout: float, memory_limit_bytes: Optional[int]
) -> Dict[str, Any]:
    """ProcessPoolExecutor worker: guard + exec with an in-process time limit.

    Unlike the per-call `check_correctness`, the pool reuses workers, so we rely
    on the in-child SIGALRM (`_time_limit`) plus the parent's `future.result`
    deadline rather than terminating the process per call.
    """
    reliability_guard(memory_limit_bytes)
    try:
        exec_globals: Dict[str, Any] = {}
        with _swallow_io():
            with _time_limit(timeout):
                exec(program, exec_globals)
        return {"passed": True, "diagnostic": "passed", "detail": ""}
    except TimeoutException:
        return {"passed": False, "diagnostic": "timeout", "detail": "time limit exceeded"}
    except AssertionError:
        return {"passed": False, "diagnostic": "failed", "detail": "assertion failed"}
    except BaseException as exc:  # noqa: BLE001
        return {
            "passed": False, "diagnostic": "exec_error",
            "detail": f"{type(exc).__name__}: {exc}",
        }
