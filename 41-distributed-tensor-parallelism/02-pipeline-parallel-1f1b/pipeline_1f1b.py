from __future__ import annotations
"""
1F1B Pipeline Parallelism Scheduler (Narayanan et al., Megatron-LM 2021).

When models are too deep to fit on one GPU, Pipeline Parallelism (PP) assigns
consecutive transformer layers to different stages (e.g., Stage 0 to Stage P-1).

Naive Pipeline Execution (GPipe):
- Runs all forward micro-batches, then all backward micro-batches.
- Suffers from memory explosion: must store activations for ALL M micro-batches simultaneously.

1F1B (One-Forward-One-Backward) Schedule:
- Warmup: Stage p runs (P - p) forward micro-batches.
- Steady State: Alternates 1 Forward step with 1 Backward step.
- Cooldown: Drains remaining backward micro-batches.
- Memory Cap: Peak activation memory is bounded to P micro-batches regardless of M!
- Bubble Fraction: F_bubble = (P - 1) / M
"""
from typing import Dict, List, Tuple
import dataclasses


@dataclasses.dataclass
class PipelineStepEvent:
    stage_id: int
    step_type: str        # 'FORWARD' or 'BACKWARD'
    microbatch_id: int


class Pipeline1F1BScheduler:
    """
    Simulates 1F1B pipeline parallel scheduling for P pipeline stages and M micro-batches.
    """
    def __init__(self, num_stages: int = 4, num_microbatches: int = 8):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

    @property
    def bubble_fraction(self) -> float:
        """Calculates theoretical idle bubble fraction: (P - 1) / M"""
        return (self.num_stages - 1) / self.num_microbatches

    def generate_stage_schedule(self, stage_id: int) -> List[PipelineStepEvent]:
        """
        Generates 1F1B execution trace for a given pipeline stage rank.
        """
        schedule = []
        P = self.num_stages
        M = self.num_microbatches

        num_warmup = min(P - stage_id, M)
        num_1f1b = M - num_warmup

        # 1. Warmup Forwards
        for mb in range(num_warmup):
            schedule.append(PipelineStepEvent(stage_id, "FORWARD", mb))

        # 2. Steady State 1F1B (Alternating Forward and Backward)
        for i in range(num_1f1b):
            fwd_mb = num_warmup + i
            bwd_mb = i
            schedule.append(PipelineStepEvent(stage_id, "FORWARD", fwd_mb))
            schedule.append(PipelineStepEvent(stage_id, "BACKWARD", bwd_mb))

        # 3. Cooldown Backwards
        for i in range(num_warmup):
            bwd_mb = num_1f1b + i
            schedule.append(PipelineStepEvent(stage_id, "BACKWARD", bwd_mb))

        return schedule


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🔄 1F1B PIPELINE PARALLELISM SCHEDULER (Megatron-LM) ===\n")

    P = 4 # 4 Pipeline Stages
    M = 8 # 8 Micro-batches
    scheduler = Pipeline1F1BScheduler(num_stages=P, num_microbatches=M)

    print(f"Configuration: {P} Pipeline Stages, {M} Micro-batches")
    print(f"Pipeline Bubble Fraction: {scheduler.bubble_fraction:.1%}\n")

    for stage in range(P):
        sched = scheduler.generate_stage_schedule(stage)
        trace_str = " -> ".join(f"{ev.step_type[0]}(mb{ev.microbatch_id})" for ev in sched)
        print(f"Stage {stage} Schedule ({len(sched)} steps):")
        print(f"  {trace_str}\n")

    print("Takeaway: 1F1B caps peak GPU activation memory to P micro-batches while keeping pipeline bubbles small!")
