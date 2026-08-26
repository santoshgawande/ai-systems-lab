from __future__ import annotations
"""
Process Reward Model (PRM) Step Verifier (Lightman et al., OpenAI 2023).

Outcome Reward Models (ORM) only score the final answer (pass/fail). If a 10-step math
derivation makes a subtle sign error on step 3 but coincidentally lands on the right number,
ORM rewards the faulty reasoning.

Process Reward Models (PRM) grade every INDIVIDUAL reasoning step:
- Identifies the exact step where hallucinations or logical fallacies begin.
- Powers Best-of-N inference search by filtering out faulty branches early.
- Dramatically increases mathematical and algorithmic problem-solving accuracy.
"""
from typing import Dict, List, Optional, Tuple
import dataclasses


@dataclasses.dataclass
class StepEvaluation:
    step_num: int
    text: str
    prm_score: float # Probability step is mathematically/logically sound [0.0, 1.0]
    is_valid: bool


@dataclasses.dataclass
class ReasoningTrajectory:
    trajectory_id: str
    steps: List[str]
    final_answer: str


class ProcessRewardVerifier:
    """
    Evaluates step-by-step reasoning trajectories and ranks candidate solutions.
    """
    def __init__(self, step_threshold: float = 0.65):
        self.step_threshold = step_threshold

    def evaluate_step(self, step_text: str, step_num: int) -> StepEvaluation:
        """
        Mock PRM heuristic evaluator:
        Flags mathematical contradictions, division by zero, or invalid syntax.
        """
        score = 0.95
        text_lower = step_text.lower()

        # Heuristic penalties for common logical flaws
        if "/ 0" in text_lower or "divided by zero" in text_lower:
            score = 0.05
        elif "therefore 1 = 2" in text_lower or "2 = 3" in text_lower:
            score = 0.02
        elif "assume without proof" in text_lower:
            score = 0.50
        elif "it follows trivially" in text_lower and len(step_text.split()) < 4:
            score = 0.40

        is_valid = score >= self.step_threshold
        return StepEvaluation(step_num=step_num, text=step_text, prm_score=score, is_valid=is_valid)

    def evaluate_trajectory(self, trajectory: ReasoningTrajectory) -> Tuple[List[StepEvaluation], float, Optional[int]]:
        """
        Evaluates all steps in a trajectory.
        Returns:
            (step_evaluations, trajectory_prm_score, first_error_step_index)
        """
        evaluations: List[StepEvaluation] = []
        first_error_step = None
        product_score = 1.0

        for i, step_text in enumerate(trajectory.steps, 1):
            eval_res = self.evaluate_step(step_text, i)
            evaluations.append(eval_res)
            product_score *= eval_res.prm_score
            if not eval_res.is_valid and first_error_step is None:
                first_error_step = i

        return evaluations, product_score, first_error_step

    def select_best_trajectory(self, candidates: List[ReasoningTrajectory]) -> Tuple[ReasoningTrajectory, float]:
        """
        Best-of-N selection using PRM step score product.
        """
        best_traj = None
        best_score = -1.0

        for traj in candidates:
            _, score, first_err = self.evaluate_trajectory(traj)
            # Heavy penalty if any step was invalid
            if first_err is not None:
                score *= 0.1
            if score > best_score:
                best_score = score
                best_traj = traj

        return best_traj, best_score


# ─── Interactive Demonstration ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 🔍 PROCESS REWARD MODEL (PRM) STEP VERIFIER ===\n")

    verifier = ProcessRewardVerifier(step_threshold=0.65)

    # Two candidate solutions for: "Solve for x: 2x + 4 = 10"
    traj_a = ReasoningTrajectory(
        trajectory_id="Path_A (Flawed Step 2)",
        steps=[
            "Step 1: Subtract 4 from both sides: 2x = 6.",
            "Step 2: Divide both sides by 0: x = 6 / 0.",  # Flaw!
            "Step 3: Therefore x = 3."
        ],
        final_answer="x = 3"
    )

    traj_b = ReasoningTrajectory(
        trajectory_id="Path_B (Sound Logic)",
        steps=[
            "Step 1: Subtract 4 from both sides: 2x = 6.",
            "Step 2: Divide both sides by 2: x = 6 / 2.",
            "Step 3: Simplify arithmetic: x = 3."
        ],
        final_answer="x = 3"
    )

    print("Evaluating Path A:")
    evals_a, score_a, err_a = verifier.evaluate_trajectory(traj_a)
    for e in evals_a:
        status = "✅ VALID" if e.is_valid else "❌ FLAW DETECTED"
        print(f"  [{e.step_num}] {status} (score={e.prm_score:.2f}) | {e.text}")
    print(f"  First Error Detected at Step: {err_a}\n")

    print("Evaluating Path B:")
    evals_b, score_b, err_b = verifier.evaluate_trajectory(traj_b)
    for e in evals_b:
        status = "✅ VALID" if e.is_valid else "❌ FLAW DETECTED"
        print(f"  [{e.step_num}] {status} (score={e.prm_score:.2f}) | {e.text}")
    print(f"  First Error Detected at Step: {err_b}\n")

    best, b_score = verifier.select_best_trajectory([traj_a, traj_b])
    print(f"🏆 Best-of-N Selected Trajectory: {best.trajectory_id} (Score: {b_score:.4f})")
    print("Takeaway: PRMs catch flawed reasoning even when the final answer accidentally matches!")
