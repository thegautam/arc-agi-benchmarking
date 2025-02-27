import argparse
from typing import List, Tuple
from pathlib import Path
from typing import List, Tuple, Dict
import json

class ARCScorer:
    def __init__(self, task_dir: str, submission_dir: str, print_logs: bool = False, results_dir: str = None):
        self.task_dir = Path(task_dir)
        self.submission_dir = Path(submission_dir)
        self.print_logs = print_logs
        self.solutions = self._load_solutions()
        self.results_dir = Path(results_dir) if results_dir else None
        self.total_cost = 0.0
        self.total_attempts = 0

    def print_log(self, message: str) -> None:
        if self.print_logs:
            print(message)

    def _load_solutions(self) -> Dict[str, List[List[List[int]]]]:
        solutions = {}
        for solution_file in self.task_dir.glob('*.json'):
            task_id = solution_file.stem
            with solution_file.open() as f:
                solutions[task_id] = json.load(f)
        return solutions

    def score_task(self, task_id: str, submission_path: Path) -> Tuple[float, float, int]:
        """
        Scores a single task submission against the solutions.
        Returns (task_score, task_cost, num_attempts)
        """
        with submission_path.open() as f:
            task_submission = json.load(f)

        task_score = 0
        num_pairs = len(task_submission)
        task_cost = 0.0
        num_attempts = 0

        for pair_index, pair_attempts in enumerate(task_submission):
            # Count all attempts in this pair, regardless of whether we process them all
            num_attempts += len(pair_attempts)
            
            # Calculate costs for all attempts upfront
            for attempt_key in pair_attempts:
                if 'metadata' in pair_attempts[attempt_key] and 'cost' in pair_attempts[attempt_key]['metadata']:
                    attempt_cost = pair_attempts[attempt_key]['metadata']['cost'].get('total_cost', 0.0)
                    task_cost += attempt_cost
            
            pair_correct = False
            for attempt in pair_attempts:
                if pair_attempts[attempt] == []:
                    self.print_log(f"    No prediction for {task_id}, pair {pair_index}, attempt {attempt}")
                    continue

                if pair_attempts[attempt]['answer'] == self.solutions[task_id]['test'][pair_index]['output']:
                    pair_correct = True
                    break

            if pair_correct:
                task_score += 1

        return task_score / num_pairs, task_cost, num_attempts

    def score_submission(self) -> Tuple[float, int]:
        """
        Read a submission from file, score it, then return the score
        """
        self.print_log(f"Scoring {self.submission_dir}\n")

        total_score = 0
        total_tasks = 0
        task_results = {}

        for submission_file in self.submission_dir.glob('*.json'):
            if submission_file.name == 'results.json':
                continue
            task_id = submission_file.stem
            task_score, task_cost, num_attempts = self.score_task(task_id, submission_file)
            
            total_score += task_score
            total_tasks += 1
            self.total_cost += task_cost
            self.total_attempts += num_attempts
            
            task_results[task_id] = {
                "score": task_score,
                "cost": task_cost,
                "attempts": num_attempts
            }

        # Calculate average costs
        avg_cost_per_task = self.total_cost / total_tasks if total_tasks > 0 else 0
        avg_cost_per_attempt = self.total_cost / self.total_attempts if self.total_attempts > 0 else 0

        # Only save results if results_dir is provided
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            results_file = self.results_dir / "results.json"
            self.print_log(f"Saving results to {results_file}")
            with results_file.open("w") as f:
                json.dump({
                    "score": total_score,
                    "total_tasks": total_tasks,
                    "total_cost": self.total_cost,
                    "total_attempts": self.total_attempts,
                    "avg_cost_per_task": avg_cost_per_task,
                    "avg_cost_per_attempt": avg_cost_per_attempt,
                    "task_results": task_results
                }, f)

        # Calculate and print percentage score
        percentage_score = (total_score / total_tasks * 100) if total_tasks > 0 else 0
        print(f"\nFinal Score: {percentage_score:.2f}% ({total_score:.2f}/{total_tasks})")
        print(f"Total Cost: ${self.total_cost:.4f}")
        print(f"Average Cost per Task: ${avg_cost_per_task:.4f}")
        print(f"Average Cost per Attempt: ${avg_cost_per_attempt:.4f}")
        print(f"Total Attempts: {self.total_attempts}")

        return total_score, total_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Tester")

    parser.add_argument("--task_dir", type=str, help="Task directory which contains the full tasks (including solutions)")
    parser.add_argument("--submission_dir", type=str, help="Submission directory which contains the submissions to score")
    parser.add_argument("--print_logs", action="store_true", help="Printing logs to console (default: False)")
    parser.add_argument("--results_dir", type=str, help="Results directory to save the results (default: None)")

    args = parser.parse_args()

    arc_scorer = ARCScorer(
        task_dir=args.task_dir,
        submission_dir=args.submission_dir,
        print_logs=args.print_logs,
        results_dir=args.results_dir
    )

    arc_scorer.score_submission()