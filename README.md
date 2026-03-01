# R1-Zero-Qwen3-235B: RL-Only Physics Reasoning

R1-Zero applies reinforcement learning **directly** to a pre-trained LLM via RL. The model learns to solve university-level physics problems through trial-and-error with an EED-based reward signal.

| | |
|---|---|
| **Base model** | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| **Method** | PPO with LoRA (rank 32), no prior SFT |
| **Training data** | ~450 physics problems (text-only) |
| **Test data** | 98 problems from *200 Puzzling Physics Problems* |

---

## Training Curves

Training runs for 29 steps (1 epoch over ~450 problems). Each step samples 16 rollouts per problem across a batch of 16 problems (256 episodes total).

### R1-Zero: EED Score, Reward, and Correct Fraction

![R1-Zero training curves](/plots/r1zero_training.png)

Key observations:
- EED score and correct fraction **increase steadily** through training, with high per-batch variance due to varying problem difficulty.
- Format compliance reaches ~100% within a few steps -- the format bonus is effective.

---

## Evaluation Results

**Test set**: 98 text-only physics problems from *200 Puzzling Physics Problems* (P200).
Evaluation uses temperature=0.6, top_p=0.95, max_tokens=16384.

| Model | Accuracy |
|-------|----------------------------|
| **R1-Zero (ckpt 20)** | **~68%** |
| Baseline (Qwen3-235B-A22B-Instruct-2507) | ~57% |

R1-Zero improves over the baseline by **+11pp** in score.
