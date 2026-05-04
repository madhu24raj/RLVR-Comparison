# Memory Optimization Spec for Cluster-Scale PPO Training

This document provides exact memory calculations, dtype guidance, and
optimization strategies for scaling the PPO pipeline from 0.5B (local dev)
to 8B (cluster target) and 70B (aspirational, multi-node).

---

## 1. Current Memory Profile

### 1.1 What the code does today (updated 2026-04-08)

| Issue | Location | Status |
|-------|----------|--------|
| float32 everywhere | `load_ppo_trainer` | **Fixed** -- auto dtype: bf16 on GPU, fp32 on CPU |
| No gradient checkpointing | `load_ppo_trainer` | **Fixed** -- `config.gradient_checkpointing` flag |
| No mixed precision | `_batched_sequence_log_probs` | **Fixed** -- logits upcast to fp32 for `log_softmax` |
| Per-sample forward passes | `_policy_log_probs` | **Fixed** -- batched via `_batched_sequence_log_probs` |
| Rollout buffer on CPU as Python lists | `Rollout.full_ids` | Open -- `List[int]` is fine for current batch sizes |

### 1.2 Memory components

For any model with P parameters, training memory breaks down as:

```
Component                      Formula (bytes)
─────────────────────────────────────────────────
Model weights                  P × bytes_per_param
Optimizer states (AdamW)       P × 8  (m + v in fp32 = 2 × 4 bytes per param)
  - with bf16 weights          P × 8  (master weights + m + v, all fp32)
Gradients                      P × bytes_per_param
Activations (forward)          f(B, S, H, L)  — see Section 4
Rollout buffer                 see Section 5
KV cache (generation only)     2 × L × H × S × bytes_per_param × B
─────────────────────────────────────────────────
```

### 1.3 Model-size reference table (weights + optimizer + gradients only)

Bytes per param: fp32 = 4, bf16 = 2, int8 = 1, int4 = 0.5.

AdamW always stores m and v in fp32 (8 bytes/param). When training in bf16
the optimizer also keeps fp32 master weights (+4 bytes/param), totaling 12
bytes/param for optimizer state. Gradients match the training dtype.

| Model | Params | **fp32 training** | **bf16 training** | **Inference only (bf16)** |
|-------|--------|-------------------|-------------------|--------------------------|
| | | W + Opt + Grad | W + Opt + Grad | W only |
| Qwen2.5-0.5B | 0.5B | 0.5×(4+8+4) = **8 GB** | 0.5×(2+12+2) = **8 GB** | **1 GB** |
| Llama-3-8B | 8B | 8×(4+8+4) = **128 GB** | 8×(2+12+2) = **128 GB** | **16 GB** |
| Llama-3-70B | 70B | 70×16 = **1120 GB** | 70×16 = **1120 GB** | **140 GB** |

> Key insight: bf16 does NOT save optimizer memory for full fine-tuning
> because AdamW states must remain in fp32. The win is in activation memory
> (2x) and compute throughput (2x on A100/H100 tensor cores).

### 1.4 Total VRAM budget (weights + optimizer + gradients + activations)

Activation estimates use B=16, S=768, values from Section 4.

| Model | fp32 no-GC | bf16 no-GC | bf16 + GC | bf16 + GC + LoRA r=16 |
|-------|-----------|-----------|----------|----------------------|
| 0.5B  | ~10 GB    | ~10 GB    | ~9 GB    | ~3 GB                |
| 8B    | ~180 GB   | ~155 GB   | ~135 GB  | ~25 GB               |
| 70B   | ~1.5 TB   | ~1.3 TB   | ~1.15 TB | ~170 GB              |

GC = gradient checkpointing. LoRA trains only ~0.5% of params.

### 1.5 GPU selection guide

| Model | Minimum viable config | Recommended config |
|-------|----------------------|-------------------|
| 0.5B (dev) | 1x RTX 3090 24GB (fp32) | 1x RTX 4090 24GB (bf16) |
| 8B (cluster) | 1x A100 80GB (bf16 + GC) | 2x A100 80GB (FSDP, bf16 + GC) |
| 8B + LoRA | 1x A100 40GB (bf16 + GC) | 1x A100 80GB (bf16 + GC) |
| 70B | 4x A100 80GB (FSDP, bf16 + GC) | 8x H100 80GB (FSDP, bf16 + GC) |
| 70B + QLoRA | 2x A100 80GB (4-bit + GC) | 4x A100 80GB |

---

## 2. Dtype Optimization

### 2.1 Where to use bf16 vs fp32

| Operation | Recommended dtype | Rationale |
|-----------|------------------|-----------|
| Model weights (forward) | **bf16** | 2x throughput on tensor cores; bf16 has same dynamic range as fp32 |
| Model weights (optimizer master) | **fp32** | AdamW accumulates small updates; bf16 rounds them away |
| `log_softmax` in `_sequence_log_prob` | **fp32** | See 2.2 |
| `gather` + `sum` for token log-probs | **fp32** | Accumulated sum over 256-512 tokens needs precision |
| Advantage computation | **fp32** | Already float32 in `compute_advantages`; keep it |
| KL divergence | **fp32** | Difference of two log-probs; subtraction magnifies relative error |
| Critic forward/backward | **fp32** | Tiny network (230K-5M params); zero memory cost; stability benefit |
| PPO ratio `exp(log_ratio)` | **fp32** | Exponentiation amplifies errors; clipping helps but fp32 is safer |
| Rollout buffer scalars | **fp32** | Already float32; negligible memory |

### 2.2 Log-prob precision analysis

`_sequence_log_prob` computes:
```python
log_probs = torch.log_softmax(outputs.logits, dim=-1)  # [1, L, V]
token_lp = response_log_probs.gather(2, response_ids.unsqueeze(-1))
return token_lp.sum(dim=-1)
```

**Problem with bf16 `log_softmax`:**
- bf16 has only 7 bits of mantissa (vs 23 for fp32).
- `log_softmax` = `log(exp(x_i) / sum(exp(x_j)))` = `x_i - log(sum(exp(x_j)))`.
- The subtraction can cancel significant digits when `x_i` is close to `logsumexp`.
- Per-token error of ~1e-3 accumulates over R=256 tokens to ~0.25 total error.
- The PPO ratio `exp(new_lp - old_lp)` then has ~28% error, enough to destabilise training.

**Solution: upcast logits before `log_softmax`:**
```python
logits_f32 = outputs.logits.float()  # upcast from bf16
log_probs = torch.log_softmax(logits_f32, dim=-1)
```
This is the standard approach used by TRL, OpenRLHF, and DeepSpeed-Chat.
The upcast is free in compute (just a type conversion) and the fp32 tensor
is transient (freed after gather).

### 2.3 Critic: keep in fp32

The critic is tiny relative to the policy:
- Small MLP: H=896 -> 256 -> 1 = ~230K params = **0.9 MB** in fp32
- Large MLP: H=4096 -> 8192 -> 8192 -> 1 = ~100M params = **400 MB** in fp32

Even the largest critic is <0.5% of an 8B model. Keeping it in fp32:
- Eliminates value prediction instability
- Has zero meaningful memory impact
- Simplifies the training loop (no mixed-precision handling for critic)

### 2.4 Implementation pattern

```python
def load_ppo_trainer(config: PPOConfig, device: torch.device) -> PPOTrainer:
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    # Critic stays in fp32 regardless
    critic = build_critic(config.critic_capacity, hidden_size)
    critic = critic.to(device)  # defaults to fp32

    ...
```

For the forward pass with log-prob computation:
```python
def _sequence_log_prob(self, input_ids, prompt_len):
    outputs = self.model(input_ids=input_ids, use_cache=False)
    # Upcast to fp32 for numerically stable log_softmax
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    ...
```

**Do NOT use `torch.cuda.amp.autocast`** for the PPO update. The forward
pass is already in bf16 natively (model loaded in bf16). The log-prob
computation needs explicit fp32 upcast. Autocast would fight against
explicit casts and add complexity with no benefit.

Use autocast only if mixed-precision is needed within a single forward
pass (e.g., some layers in fp32, others in bf16). For our pipeline,
the model is uniformly bf16, and post-model ops are explicitly fp32.

---

## 3. Gradient Checkpointing

### 3.1 How to enable

```python
if device.type == "cuda":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
```

Call this in `load_ppo_trainer` after loading the model, before `.to(device)`.

### 3.2 Reentrant vs non-reentrant

| Aspect | Reentrant (`use_reentrant=True`) | Non-reentrant (`use_reentrant=False`) |
|--------|----------------------------------|---------------------------------------|
| PyTorch default | Yes (legacy) | No (recommended since PyTorch 2.1) |
| Works with `torch.no_grad()` | Broken — silently incorrect grads | Correct |
| Works with `output_hidden_states=True` | May produce incorrect gradients | Correct |
| Supports non-tensor inputs | No — must explicitly handle | Yes |
| Future-proof | Deprecated in PyTorch 2.4+ | Recommended going forward |

**Use non-reentrant.** The reentrant variant has known bugs with
`output_hidden_states=True` (needed for critic hidden state extraction)
and with mixed `torch.no_grad()` / grad contexts (used in `_critic_forward`).

### 3.3 Impact on `_policy_log_probs`

`_policy_log_probs` is the only with-grad forward pass in the PPO update.
It calls `self.model(input_ids=full_ids, use_cache=False)` — this is the
call that builds the full backward graph and consumes most activation memory.

With gradient checkpointing enabled:
- Each transformer layer's activations are discarded after the forward pass
- During backward, each layer re-runs its forward to recompute activations
- Memory: O(sqrt(L)) instead of O(L) for L layers
- Compute: ~33% overhead (each layer is computed twice: once forward, once during backward)

**Compatibility with `use_cache=False`:** Required. Gradient checkpointing
and KV caching are mutually exclusive. The code already sets `use_cache=False`
for all non-generation forward passes, so this is compatible.

### 3.4 Impact on `output_hidden_states=True`

The `_critic_forward` method calls `self.model(..., output_hidden_states=True)`
to extract hidden states for the critic. With non-reentrant gradient
checkpointing, this works correctly. The hidden states are recomputed
during backward as needed.

However, `_critic_forward` runs under `torch.no_grad()` for the model
(only the critic head has gradients). Gradient checkpointing has no effect
under `torch.no_grad()` — it simply runs a normal forward pass. This is
correct behavior: no activations need to be saved because there is no
backward pass through the model.

### 3.5 Interaction with P2 (combined forward pass)

After applying P2 from `performance.md` (merging `_critic_forward` and
`_policy_log_probs` into a single forward pass), the combined pass runs
WITH gradients and WITH `output_hidden_states=True`. Non-reentrant
gradient checkpointing handles this correctly. The hidden states at the
prompt boundary (for critic input) are available during forward, and the
backward pass recomputes layer activations as needed.

---

## 4. Activation Memory During PPO Update

### 4.1 Per-layer activation memory formula

For a standard transformer layer in mixed precision (bf16 forward):

```
Per layer = s × b × h × (34 + 5 × a × s / h) bytes
         ≈ 34sbh + 5abs²    bytes

Where:
  s = sequence length
  b = batch size (per forward call; =1 in current per-sample loop)
  h = hidden size
  a = number of attention heads
  L = number of layers
```

The 34sbh term covers: input activations, QKV projections, output
projection, FFN activations (2x for up/down + gelu input), layer norm
inputs. The 5abs² term covers the attention score matrix.

Total activation memory = L × per_layer.

### 4.2 Concrete calculations for 8B model (Llama-3-8B)

Model parameters: H=4096, L=32, A=32, V=128256.

**Scenario: B=1 (current per-sample loop), S=768**

```
Per layer  = 768 × 1 × 4096 × (34 + 5 × 32 × 768 / 4096)
           = 3,145,728 × (34 + 30)
           = 3,145,728 × 64
           = 201 MB per layer

Total (32 layers) = 201 × 32 = 6.4 GB
Logits tensor     = 1 × 768 × 128256 × 2 = 188 MB (bf16)
                                            376 MB (fp32 after upcast)
─────────────────────────────────
Total activations ≈ 6.8 GB (per sample, with grad)
```

**Scenario: B=16 (batched after P1 fix), S=768**

```
Per layer  = 768 × 16 × 4096 × 64 = 3.2 GB per layer
Total (32 layers) = 3.2 × 32 = 102 GB
Logits tensor     = 16 × 768 × 128256 × 2 = 3.0 GB
─────────────────────────────────
Total activations ≈ 105 GB  ← does not fit on A100 80GB
```

### 4.3 With gradient checkpointing

Gradient checkpointing stores only the input to each checkpointed segment
(typically one transformer layer = one segment). The per-segment stored
activation is just the layer input: `s × b × h × 2` bytes (bf16).

```
Stored per layer    = 768 × 16 × 4096 × 2 = 96 MB
Total (32 layers)   = 96 × 32 = 3.1 GB
Peak recompute      = 1 layer worth of activations = 3.2 GB
─────────────────────────────────
Total activations ≈ 6.3 GB  ← fits on A100 80GB
```

### 4.4 With micro-batching (gradient accumulation)

Instead of processing B=16 in one forward pass, split into micro-batches
of size `mbs` and accumulate gradients:

```
Effective activation memory = (activations for mbs samples) × 1
                            + gradient accumulation over B/mbs steps

For mbs=1 (current code): 6.8 GB activations (per sample)
For mbs=4:                27.2 GB activations
For mbs=16:               105 GB activations (full batch)
```

**Recommendation for 8B on 1x A100 80GB:**
- bf16 + gradient checkpointing + micro-batch size 2-4
- Total: 16 GB (weights) + ~6 GB (activations) + ~48 GB (optimizer + grads) = ~70 GB

### 4.5 Summary table: activation memory (bf16, S=768)

| Model | B=1, no GC | B=1, GC | B=16, no GC | B=16, GC | B=16, GC, mbs=2 |
|-------|-----------|---------|------------|---------|----------------|
| 0.5B (H=896, L=24) | 0.4 GB | 0.05 GB | 6.4 GB | 0.8 GB | 0.1 GB |
| 8B (H=4096, L=32) | 6.8 GB | 0.4 GB | 105 GB | 6.3 GB | 0.8 GB |
| 70B (H=8192, L=80) | 34 GB | 1.6 GB | 540 GB | 25 GB | 3.2 GB |

GC = gradient checkpointing. mbs = micro-batch size for gradient accumulation.

---

## 5. Rollout Buffer Memory

### 5.1 Current storage format

```python
@dataclass
class Rollout:
    prompt: str                 # Python string
    completion: str             # Python string
    reward: float               # Python float (8 bytes)
    old_log_prob: float         # Python float (8 bytes)
    value: float                # Python float (8 bytes)
    full_ids: List[int]         # Python list of ints (28 bytes per int on 64-bit Python)
    prompt_len: int             # Python int (28 bytes)
```

### 5.2 Memory calculation for B=16, max_seq=768

```
Per rollout:
  full_ids:  768 Python ints × 28 bytes = 21.5 KB
  strings:   ~1-2 KB (prompt + completion)
  scalars:   ~100 bytes
  ─────────────────────────────
  Total: ~23 KB per rollout

Batch of 16: 16 × 23 KB = 368 KB
```

This is negligible (<1 MB). The rollout buffer is not a memory concern.

### 5.3 Should `full_ids` be a padded tensor?

**Current (Python lists):**
- Pro: No wasted memory on padding; lives on CPU; no GPU memory
- Con: Requires per-sample `torch.tensor()` conversion in `_policy_log_probs`

**Alternative (padded GPU tensor):**
- Pro: Enables batched forward passes after P1 fix
- Con: Wastes memory on padding (variable-length sequences)

**Recommendation:** When implementing batched forward passes (P1, P2 from
`performance.md`), convert `full_ids` to a padded tensor at the start of
`ppo_update`, not at rollout storage time. This keeps the buffer lean and
avoids holding GPU memory between training steps.

```python
def _pad_full_ids(self, batch: RolloutBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad full_ids to a tensor with attention mask."""
    max_len = max(len(r.full_ids) for r in batch.rollouts)
    padded = torch.full((len(batch.rollouts), max_len),
                        self.tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(batch.rollouts), max_len, dtype=torch.bool)
    for i, r in enumerate(batch.rollouts):
        L = len(r.full_ids)
        padded[i, :L] = torch.tensor(r.full_ids, dtype=torch.long)
        mask[i, :L] = True
    return padded.to(self.device), mask.to(self.device)
```

For B=16, S=768, this tensor is `16 × 768 × 8 = 98 KB` (int64) or
`16 × 768 × 4 = 49 KB` (int32). Still negligible.

---

## 6. Model Loading Optimization

### 6.1 Quantized inference (bitsandbytes)

During rollout generation, the model runs in `@torch.no_grad()` mode.
This is an inference-only phase where quantization can reduce memory
without affecting gradient computation.

| Method | Memory for 8B | Quality impact | Compatibility |
|--------|--------------|----------------|---------------|
| bf16 (baseline) | 16 GB | None | Full |
| load_in_8bit | 8 GB | Minimal (<0.5% accuracy) | No training; inference only |
| load_in_4bit (NF4) | 4 GB | Small (~1-2% accuracy) | No training; inference only |

**Problem:** The PPO pipeline uses the SAME model for both generation
(inference) and policy gradient updates (training). Quantized models
cannot compute gradients through the quantized weights.

**Options:**
1. **Separate generation and training models** — load a quantized copy for
   generation and a bf16 copy for training. Doubles memory. Not recommended.
2. **LoRA/QLoRA** — see 6.2.
3. **Keep bf16 for everything** — simplest; recommended for 8B on A100 80GB.

### 6.2 LoRA / QLoRA for PPO

LoRA adds small low-rank adapter matrices (A, B) to selected weight
matrices. Only A and B are trained; the base model is frozen.

**Memory savings with LoRA (rank=16, applied to q_proj, v_proj):**

```
Trainable params = 2 × rank × hidden × 2 × n_layers  (q and v projections)
                 = 2 × 16 × 4096 × 2 × 32
                 = 8.4M params (0.1% of 8B)

Optimizer states  = 8.4M × 12 = 100 MB  (vs 96 GB for full fine-tuning)
Gradients         = 8.4M × 2 = 17 MB    (vs 16 GB)
Base model        = 8B × 2 = 16 GB      (frozen, bf16, no optimizer states)
─────────────────────────────────
Total: ~16.1 GB + activations   (vs ~128 GB + activations)
```

**QLoRA** additionally quantizes the frozen base model to 4-bit:
```
Base model = 8B × 0.5 = 4 GB (NF4)
Total: ~4.1 GB + activations
```

**How LoRA interacts with the PPO update:**

1. **Policy gradient computation:** `_policy_log_probs` runs a forward pass
   through the full model (base + adapters). Gradients flow only through
   the LoRA adapter weights. The `log_softmax` → `gather` → `sum` pipeline
   works identically — the output logits are influenced by the adapters.

2. **Optimizer:** Only updates LoRA parameters. AdamW states are tiny.

3. **PPO ratio:** `ratio = exp(new_lp - old_lp)` works the same. The
   adapters change the model's output distribution, so `new_lp` differs
   from `old_lp` as the adapters are updated.

4. **Critic:** The critic receives hidden states from the model (with
   adapters active). As adapters change, hidden state distributions shift.
   The critic adapts via its own gradient updates.

5. **Reference model KL:** With LoRA, the reference model is simply the
   base model WITHOUT adapters (set adapters to zero / disable). This is
   essentially free — no extra memory needed.

**Caveats for LoRA + PPO:**
- LoRA's low rank limits the expressiveness of policy updates. For RLVR
  with sparse binary rewards, this is usually sufficient (the policy
  signal is itself low-rank).
- Generation quality may be slightly lower than full fine-tuning.
- LoRA works well with `peft` library integration; HuggingFace TRL's
  `PPOTrainer` has native LoRA support.

### 6.3 Implementation sketch for LoRA

```python
from peft import get_peft_model, LoraConfig, TaskType

def load_ppo_trainer_lora(config: PPOConfig, device: torch.device) -> PPOTrainer:
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # → trainable params: 8.4M || all params: 8.0B || trainable%: 0.10%

    model.to(device)
    if device.type == "cuda":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Optimizer only gets LoRA params
    # (filter for requires_grad=True in PPOTrainer.__init__)
    ...
```

For QLoRA, add `load_in_4bit=True` and `bnb_4bit_compute_dtype=torch.bfloat16`
to `from_pretrained`.

---

## 7. Complete Memory Budget Tables

### 7.1 Qwen2.5-0.5B (H=896, L=24, A=14, V=151936)

B=16, S=768. All values in GB.

| Component | fp32 | bf16 | bf16 + GC | bf16 + GC + LoRA |
|-----------|------|------|-----------|------------------|
| Weights | 2.0 | 1.0 | 1.0 | 1.0 (frozen) + ~0.001 |
| Optimizer | 4.0 | 6.0 | 6.0 | 0.01 |
| Gradients | 2.0 | 1.0 | 1.0 | 0.001 |
| Activations | 12.8 | 6.4 | 0.8 | 0.8 |
| Rollout buffer | <0.01 | <0.01 | <0.01 | <0.01 |
| **Total** | **~21 GB** | **~14 GB** | **~9 GB** | **~2 GB** |
| Fits on | RTX 3090 24GB | RTX 3090 24GB | RTX 3060 12GB | Any GPU |

### 7.2 Llama-3-8B (H=4096, L=32, A=32, V=128256)

B=16, S=768. All values in GB.

| Component | fp32 | bf16 | bf16 + GC | bf16 + GC + mbs=2 | bf16 + GC + LoRA |
|-----------|------|------|-----------|-------------------|------------------|
| Weights | 32 | 16 | 16 | 16 | 16 (frozen) + 0.02 |
| Optimizer | 64 | 96 | 96 | 96 | 0.1 |
| Gradients | 32 | 16 | 16 | 16 | 0.02 |
| Activations | 210 | 105 | 6.3 | 0.8 | 0.8 |
| Rollout buffer | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| **Total** | **~338 GB** | **~233 GB** | **~134 GB** | **~129 GB** | **~17 GB** |
| Fits on | N/A | N/A | 2x A100 80GB | 2x A100 80GB | 1x A100 40GB |

> Note: bf16 optimizer is larger than fp32 because AdamW stores fp32 master
> weights (4B/param) + m (4B/param) + v (4B/param) = 12 B/param, vs 8 B/param
> when the model is already fp32 (no master copy needed).

### 7.3 Llama-3-70B (H=8192, L=80, A=64, V=128256)

B=16, S=768. All values in GB.

| Component | bf16 + GC | bf16 + GC + mbs=1 | bf16 + GC + LoRA | QLoRA (4-bit) + GC |
|-----------|-----------|-------------------|------------------|--------------------|
| Weights | 140 | 140 | 140 | 35 |
| Optimizer | 840 | 840 | 0.8 | 0.8 |
| Gradients | 140 | 140 | 0.15 | 0.15 |
| Activations | 25 | 3.2 | 3.2 | 3.2 |
| **Total** | **~1145 GB** | **~1123 GB** | **~144 GB** | **~39 GB** |
| Fits on | 16x A100 80GB (FSDP) | 16x A100 80GB (FSDP) | 2x A100 80GB | 1x A100 80GB |

---

## 8. Implementation Priority for Cluster Scale (8B)

Apply these optimizations in order. Each row shows the cumulative memory.

| Step | Change | VRAM saved | Cumulative total | Can train on |
|------|--------|-----------|-----------------|--------------|
| 0 | Current (fp32, no GC) | — | ~338 GB | Nothing |
| 1 | bf16 model loading | -105 GB (activations) | ~233 GB | Nothing |
| 2 | Gradient checkpointing | -99 GB (activations) | ~134 GB | 2x A100 80GB |
| 3 | Micro-batch (mbs=2) | -5.5 GB (activations) | ~129 GB | 2x A100 80GB |
| 4 | FSDP (shard optimizer) | -64 GB per GPU | ~65 GB/GPU | 2x A100 80GB |
| **Alt** | **LoRA r=16 (instead of steps 0-4)** | **-321 GB** | **~17 GB** | **1x A100 40GB** |

### Recommended cluster configuration for 8B

**Option A — Full fine-tuning (highest quality):**
```
2x A100 80GB, FSDP ZeRO-3, bf16, gradient checkpointing, mbs=2
```

**Option B — LoRA (fastest, cheapest, nearly same quality for RLVR):**
```
1x A100 80GB, LoRA r=16, bf16, gradient checkpointing
```

---

## 9. Code Changes Summary

### 9.1 Minimal changes for 8B cluster support

**`config.py`** — add memory-related config fields:
```python
# Memory optimization
torch_dtype: str = "auto"           # "float32", "bfloat16", "auto"
gradient_checkpointing: bool = True
micro_batch_size: int = 2           # for gradient accumulation in PPO update
use_lora: bool = False
lora_rank: int = 16
lora_target_modules: List[str] = field(
    default_factory=lambda: ["q_proj", "v_proj"]
)
```

**`ppo_trainer.py`** — changes in `load_ppo_trainer`:
```python
# 1. Dtype selection
if config.torch_dtype == "auto":
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
else:
    torch_dtype = getattr(torch, config.torch_dtype)

# 2. Model loading
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch_dtype,
).to(device)

# 3. Gradient checkpointing
if config.gradient_checkpointing and device.type == "cuda":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

# 4. LoRA (optional)
if config.use_lora:
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
```

**`ppo_trainer.py`** — fp32 upcast in `_sequence_log_prob`:
```python
def _sequence_log_prob(self, input_ids, prompt_len):
    outputs = self.model(input_ids=input_ids, use_cache=False)
    # Critical: upcast to fp32 for numerical stability
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    ...
```

**`ppo_trainer.py`** — micro-batching in `_policy_log_probs`:
```python
def _policy_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
    log_probs: List[torch.Tensor] = []
    for rollout in batch.rollouts:
        full_ids = torch.tensor(
            [rollout.full_ids], dtype=torch.long, device=self.device
        )
        lp = self._sequence_log_prob(full_ids, rollout.prompt_len)
        log_probs.append(lp.squeeze(0))
    return torch.stack(log_probs)
```
The current per-sample loop already functions as mbs=1 micro-batching.
After implementing batched forward passes (P1/P2), add explicit gradient
accumulation with configurable micro-batch size.

---

## 10. Verification Checklist

Before running on the cluster, verify:

- [ ] Model loads in bf16 on GPU, fp32 on CPU
- [ ] `log_softmax` receives fp32 logits (add assert: `assert logits.dtype == torch.float32`)
- [ ] Gradient checkpointing is active (`model.is_gradient_checkpointing == True`)
- [ ] Critic remains in fp32 (`next(critic.parameters()).dtype == torch.float32`)
- [ ] KL divergence computed in fp32
- [ ] Advantage normalization in fp32
- [ ] PPO ratio clamped before exp (already done: `torch.clamp(log_ratio, -20, 20)`)
- [ ] GPU memory stays below 75 GB on A100 80GB (leave headroom for fragmentation)
- [ ] Training loss matches fp32 baseline within 5% over 50 steps (regression test)

---

## 11. Deep Memory Analysis (added 2026-04-30)

This section contains memory wins identified by a quantitative PhD-level
deep review. Every number is computed in bytes; numbers are for Llama-3-8B
unless noted.

### 11.1 8-bit AdamW (bitsandbytes) — biggest single 8B win

For 8B models, the dominant memory cost is AdamW (96 GB at fp32 for m+v
+ master). bitsandbytes' Block-wise quantized `AdamW8bit` stores m and v
in int8 (8.03 GB each) while keeping fp32 master weights (32 GB).

| Component | fp32 AdamW | 8-bit AdamW | Savings |
|-----------|-----------:|------------:|--------:|
| m (first moment) | 32 GB | 8 GB | 24 GB |
| v (second moment) | 32 GB | 8 GB | 24 GB |
| master fp32 | 32 GB | 32 GB | 0 |
| **Total** | **96 GB** | **48 GB** | **48 GB** |

Quality loss: <0.5% per the bnb paper. Compatible with FSDP. Implementation:

```python
# ppo_trainer.py:152-153 — replace
if config.optimizer_8bit:
    import bitsandbytes as bnb
    self.policy_optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=config.learning_rate
    )
else:
    self.policy_optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )
```

Add `optimizer_8bit: bool = False` to `PPOConfig`. Add
`bitsandbytes>=0.41` to `requirements.txt`. Add a unit test that
constructs both optimizer types and confirms a 1-step parameter delta
agrees within rtol=1e-3.

### 11.2 Quantize the FROZEN reference model

The frozen reference model used for `KL(pi_new || pi_ref)` (loaded at
[ppo_trainer.py:719-729](../ppo_trainer.py#L719-L729)) computes log-probs
only — never gradients. Quantization restrictions on training do not
apply.

| Quantization | Memory | KL log-prob drift | Notes |
|--------------|-------:|------------------:|-------|
| bf16 (current) | 16 GB | 0 (reference) | — |
| int8 (bnb LLM.int8) | 8 GB | ~1e-3 nats | Linear layer outliers handled |
| NF4 (bnb 4-bit) | 4 GB | ~5e-3 nats | NormalFloat4 quantization |

Implementation at [ppo_trainer.py:723-726](../ppo_trainer.py#L723-L726):

```python
quant_config = None
if config.reference_quant == "int8":
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
elif config.reference_quant == "nf4":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
reference_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, dtype=torch_dtype,
    quantization_config=quant_config,
)
```

KL drift of 1e-3 nats is negligible vs typical per-token KL of 0.01-1.0
during healthy PPO. Add `reference_quant: str = "none"` to `PPOConfig`
(values: "none" | "int8" | "nf4").

### 11.3 Checkpoint resume causes 96 GB transient peak (bug)

[checkpoint.py:128](../checkpoint.py#L128):
```python
policy_opt_state = torch.load(str(ckpt / "policy_optimizer.pt"), map_location=device, weights_only=True)
```
Loads the optimizer state directly to GPU. Then [run_e2_7.py:105](../run_e2_7.py#L105)
calls `load_state_dict(state["policy_optimizer_state_dict"])`, which copies
INTO the optimizer's parameter slots. During the copy, BOTH the loaded
tensors AND the optimizer's existing state coexist on GPU.

For 8B AdamW: peak transient = **2 × 96 = 192 GB** during resume.
Reproduces only on resume, not on fresh runs — easy to miss in CI.

**Fix:** load optimizer state to CPU, drop the dict after `load_state_dict`:

```python
# checkpoint.py:128-130 — change map_location to "cpu"
policy_opt_state = torch.load(
    str(ckpt / "policy_optimizer.pt"),
    map_location="cpu",
    weights_only=True,
)
# load_state_dict handles per-tensor placement to optimizer.param.device
```

And in run scripts after `load_state_dict`:
```python
trainer.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])
del state["policy_optimizer_state_dict"]
gc.collect()
torch.cuda.empty_cache()
```

Same applies to `critic.pt`, `critic_optimizer.pt` at lines 130, 133.

### 11.4 AdamW state allocates lazily on first `.step()`

PyTorch's `torch.optim.AdamW` does NOT pre-allocate m and v. They are
created in the first `.step()` call. This means:

1. Memory measurements at step 0 UNDER-report by 64 GB (fp32 m+v at 8B).
2. The very first `ppo_update`'s `.step()` call has a one-time +64 GB
   peak; if the rest of the budget is tight, the first step OOMs but
   the rollout doesn't.
3. `torch.cuda.memory_summary()` should be called AFTER step 1, not
   before, to capture steady-state peak.

Add to verification checklist (§10):
- [ ] Run `torch.cuda.memory_allocated() / 1e9` after step 1 for true peak.
- [ ] If first step OOMs but rollout doesn't, the cause is m+v allocation;
      reduce batch_size or switch to AdamW8bit (§11.1).

### 11.5 AdamW step transient via `foreach=True`

PyTorch 2.x defaults `torch.optim.AdamW(foreach=True)` on CUDA. The
foreach implementation allocates working tensors of size ~P during
`.step()`: at 8B that's ~16 GB transient. With `foreach=False` it's
slower but lower peak.

**Recommended:** use `fused=True` (CUDA-only) which uses a single kernel
and bypasses the transient:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    fused=True,   # CUDA-only; no transient
)
```

Add fallback for CPU: `fused=False, foreach=False` for CPU smoke.

### 11.6 `output_hidden_states=True` returns ALL layers

[ppo_trainer.py:303-308](../ppo_trainer.py#L303-L308) sets
`output_hidden_states=True` to grab the LAST layer's hidden state.
HF returns ALL `n_layers+1` hidden states. At Llama-3-8B B=16 S=768:
`33 × 96 MB = 3.17 GB` transient (bf16). Only the final 96 MB is used.

**Mitigation:** use a forward hook on the last decoder layer to capture
just what's needed:

```python
def _capture_last_hidden_hook(module, input, output):
    self._last_hidden = output[0]  # tuple

handle = self.model.model.layers[-1].register_forward_hook(_capture_last_hidden_hook)
try:
    with torch.no_grad():
        _ = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    last_hidden = self._last_hidden
finally:
    handle.remove()
```

Saves ~3 GB transient. May unlock B=24 from B=16 on A100 80GB. Adds
~1 line of complexity but is well-isolated to `_extract_last_hidden`.

### 11.7 `outputs.logits` lifetime in batched_per_token_log_probs

[shared/per_token_loss.py:75-90](../../shared/per_token_loss.py#L75-L90)
loops per-sample to avoid the [B,S,V] `log_softmax` allocation. **However,**
the underlying `outputs.logits` of shape `[B, S, V]` in bf16 is
materialized by the LM forward at line 57-59: `16 × 768 × 128256 × 2 = 3.0 GB`
(8B) or `16 × 768 × 151936 × 2 = 3.7 GB` (Qwen-0.5B-Instruct vocab).

This tensor is held until the per-sample loop exits. Under autograd (PPO
update path with grad), it is also retained for backward.

**Mitigation:** switch to `torch.nn.functional.cross_entropy(reduction='none')`
which fuses log_softmax+gather into a single kernel without materializing
the per-position log_softmax tensor:

```python
# In shared/per_token_loss.py per_sample inner: replace
lp_full = F.log_softmax(slice_logits.float(), dim=-1)  # [R, V]
token_lp = lp_full.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
# with
token_lp = -F.cross_entropy(
    slice_logits.float(), target_ids, reduction="none",
)
```

Backward then uses only the bf16 logits (already paid for). Savings:
the transient fp32 [R,V] (~196 MB per sample, peak ~3.1 GB under K=4
under grad) drops to ~0.

### 11.8 PYTORCH_CUDA_ALLOC_CONF for fragmentation

For 8B-scale runs, set the expandable allocator at process start:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This reduces fragmentation when allocations are interleaved with
generate() (large transient KV cache) and ppo_update (large optimizer
moments). Without it, expect 5-10% OOM-headroom loss on A100-80GB
under the rollout↔update cycle.

Add to `scripts/slurm_e2_7.sh` and `scripts/slurm_e2_8.sh` near the
top of the GPU branch.

### 11.9 KV cache peak during generation

Llama-3-8B (GQA: `n_kv_heads=8`, `head_dim=128`) with `max_new_tokens=384`
and `S_prompt=512`:

```
cache_per_token = 2 × 8 × 128 × 2 B = 4096 B/layer/slot
peak_KV = B × (S_prompt + T_max) × n_layers × cache_per_token
       = 16 × 896 × 32 × 4096 = 1.84 GB
```

KV cache scales LINEARLY with batch and sequence; halving the prompt
length saves ~1 GB. Setting `max_new_tokens=256` (down from 384) saves
~0.5 GB during generation.

Generate-time activations (no GC during generation, since no backward):
peak ~6 GB at the prefill step.

**Total peak during rollout:** weights (16) + KV (1.8) + activations (6)
= ~24 GB. This is BEFORE any reference/RM. With reference + RM at bf16
each: 24 + 16 + 16 = **56 GB** during rollout phase alone.

### 11.10 Updated per-rank GPU memory budget

Llama-3-8B, B=16, S_prompt=512, T=384, ref+RM+policy at bf16:

| Configuration | Peak GB | Fits 80 GB A100? |
|---------------|--------:|:-----------------|
| Default (fp32 AdamW, no quant, no GC) | 167 | NO |
| + Gradient checkpointing | 89 | Yes (tight) |
| + 8-bit AdamW | 41 | Yes |
| + Reference int8 | 33 | Yes |
| + RM int8 | 25 | Yes (comfortable) |
| + Reference NF4 + RM NF4 | 17 | Yes (very comfortable) |

The full mitigation stack (GC + 8bit-AdamW + ref-int8 + RM-int8) brings
8B + reference + RM down to **25 GB peak** — fits 1× A100 40GB. This is
the recommended target configuration for cluster runs.

## 12. Top 5 memory wins ranked by GB saved (8B)

| # | Optimization | Savings | Effort | Risk |
|---|--------------|--------:|--------|------|
| 1 | 8-bit AdamW (bnb.optim.AdamW8bit) | 48 GB | 1 hour | Low (<0.5% loss) |
| 2 | Gradient checkpointing on policy | ~99 GB activations | 0 effort (config flag) | None |
| 3 | Quantize frozen reference model (int8) | 8 GB | 1 hour | Low (frozen, no training) |
| 4 | Fuse log_softmax+gather (cross_entropy) | ~3 GB peak | 1.5 hours | Medium (numerical equivalence) |
| 5 | CPU-offload frozen models during PPO loop | 32 GB during gradient phase | 2 hours | Medium (~2 s/step PCIe overhead) |

**Stretch (deferred):** ZeRO-2 sharding of optimizer state across 4
ranks: drops AdamW from 96 → 24 GB/rank, saving 72 GB/rank.

---

## Sources

- [OpenRLHF Framework](https://github.com/OpenRLHF/OpenRLHF) — production RLHF with gradient checkpointing and FSDP
- [Transformer Math 101 (EleutherAI)](https://blog.eleuther.ai/transformer-math/) — activation memory formulas
- [Reducing Activation Recomputation in Large Transformer Models (Nvidia)](https://arxiv.org/pdf/2205.05198) — selective recomputation strategies
- [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU (HuggingFace)](https://huggingface.co/blog/trl-peft) — LoRA + RLHF integration
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html) — autocast and GradScaler reference
- [Gradient Checkpointing Guide (Python-bloggers)](https://python-bloggers.com/2024/09/mastering-gradient-checkpoints-in-pytorch-a-comprehensive-guide/) — reentrant vs non-reentrant
- [Training-Inference Mismatch via FP16 (arXiv)](https://arxiv.org/html/2510.26788v1) — bf16 precision issues in RL fine-tuning
- [HuggingFace Performance Guide](https://huggingface.co/docs/transformers/en/perf_train_gpu_one) — GPU training optimization
- [verl LoRA PPO Documentation](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) — LoRA with PPO in practice
- [Gradient Checkpointing: The Unreasonable Impact (Kaitchup)](https://kaitchup.substack.com/p/gradient-checkpointing-llms) — practical memory savings
