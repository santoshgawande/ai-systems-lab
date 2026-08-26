# 34. Reasoning Models & Test-Time Compute

Test-Time Compute (TTC) scaling and Process Reward Models (PRMs) represent the frontier of AI capabilities, shifting focus from pure pre-training scale to extended inference-time verification and deliberate chain-of-thought problem solving.

## Labs

| Lab | Name | What You Learn |
|---|---|---|
| `01-test-time-compute` | Test-Time Compute & Thinking Budgets | `<think>` parsing, dynamic compute budgets, self-reflection detection |
| `02-process-reward-verifier` | Process Reward Model (PRM) Verifier | Step-by-step supervision, early mistake localization, Best-of-N search |

## Key Concepts

- **Test-Time Compute Scaling**: Reasoning accuracy scales logarithmically with inference compute tokens (allowing a 7B model to outscore a 70B model on math).
- **Process Supervision**: Rewarding intermediate steps rather than just final answers eliminates guessing and hallucinated proofs.
- **Backtracking & Error Recovery**: Models learn to catch intermediate errors mid-thought and self-correct prior to answer emission.
