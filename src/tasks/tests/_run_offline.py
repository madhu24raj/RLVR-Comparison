"""Tiny no-pytest runner for the sandbox tests (spawn needs a real __main__ file)."""
import inspect
import os
import pathlib
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    import src.tasks.tests.test_code_exec as T

    passed = failed = 0
    for name in sorted(d for d in dir(T) if d.startswith("test_")):
        fn = getattr(T, name)
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
