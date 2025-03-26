import argparse
from typing import List, Tuple
from pathlib import Path
from typing import List, Tuple, Dict
import json
import numpy as np

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

    def calculate_pixel_similarity(self, matrix_a: List[List[int]], matrix_b: List[List[int]]) -> float:
        """
        Calculate pixel similarity between two matrices with smart alignment if necessary.
        Returns a similarity score between 0.0 and 1.0.
        """
        # First validate the input matrices before conversion
        try:
            # Check if all rows have the same length
            if not all(isinstance(row, list) for row in matrix_a) or not all(isinstance(row, list) for row in matrix_b):
                return 0.0
                
            # Check if all rows have the same length
            if len(set(len(row) for row in matrix_a)) != 1 or len(set(len(row) for row in matrix_b)) != 1:
                return 0.0
                
            # Convert to numpy arrays only after validation
            matrix_a = np.array(matrix_a, dtype=int)
            matrix_b = np.array(matrix_b, dtype=int)
            
            # Handle empty matrices
            if matrix_a.size == 0 or matrix_b.size == 0:
                return 0.0
                
        except (ValueError, TypeError):
            # Catch any conversion errors
            return 0.0
        
        # Step 1: Check dimensions
        if matrix_a.shape == matrix_b.shape:
            aligned_a, aligned_b = matrix_a, matrix_b
        else:
            # Step 2: Align matrices (when shapes differ)
            # Identify smaller and larger matrices
            if matrix_a.size <= matrix_b.size:
                matrix_small, matrix_large = matrix_a, matrix_b
                small_is_a = True
            else:
                matrix_small, matrix_large = matrix_b, matrix_a
                small_is_a = False
            
            # Check if smaller matrix can fit inside larger matrix
            if (matrix_small.shape[0] > matrix_large.shape[0] or 
                matrix_small.shape[1] > matrix_large.shape[1]):
                # If smaller matrix can't fit inside larger, use the overlapping region
                overlap_height = min(matrix_small.shape[0], matrix_large.shape[0])
                overlap_width = min(matrix_small.shape[1], matrix_large.shape[1])
                
                # Extract overlapping regions
                small_region = matrix_small[:overlap_height, :overlap_width]
                large_region = matrix_large[:overlap_height, :overlap_width]
                
                if small_is_a:
                    aligned_a, aligned_b = small_region, large_region
                else:
                    aligned_a, aligned_b = large_region, small_region
            else:
                # Normal case: smaller matrix can fit inside larger matrix
                max_matches = -1
                best_position = (0, 0)
                
                # Iterate over all possible positions
                for x in range(matrix_large.shape[0] - matrix_small.shape[0] + 1):
                    for y in range(matrix_large.shape[1] - matrix_small.shape[1] + 1):
                        current_matches = 0
                        for i in range(matrix_small.shape[0]):
                            for j in range(matrix_small.shape[1]):
                                if matrix_small[i, j] == matrix_large[i+x, j+y]:
                                    current_matches += 1
                        
                        # Keeps track of best match position
                        if current_matches > max_matches:
                            max_matches = current_matches
                            best_position = (x, y)
                
                # Crop aligned region from larger matrix
                x, y = best_position
                aligned_region = matrix_large[x:x+matrix_small.shape[0], y:y+matrix_small.shape[1]]
                
                # Ensure we assign the matrices correctly
                if small_is_a:
                    aligned_a, aligned_b = matrix_small, aligned_region
                else:
                    aligned_a, aligned_b = aligned_region, matrix_small
        
        # Verify shapes match before comparison
        assert aligned_a.shape == aligned_b.shape, f"Shapes don't match after alignment: {aligned_a.shape} vs {aligned_b.shape}"
        
        # Step 3: Calculate pixel-wise similarity
        matching_pixels = np.sum(aligned_a == aligned_b)
        total_pixels = aligned_a.size
        
        # Add size penalty when matrices have very different dimensions
        original_a = np.array(matrix_a)
        original_b = np.array(matrix_b)
        size_ratio = min(original_a.size, original_b.size) / max(original_a.size, original_b.size)
        
        # Calculate similarity with size penalty
        pixel_similarity = matching_pixels / total_pixels if total_pixels > 0 else 0.0
        final_similarity = pixel_similarity * size_ratio
        
        return final_similarity

    def score_task(self, task_id: str, submission_path: Path) -> Tuple[float, float, int]:
        """
        Scores a single task submission against the solutions.
        Returns (task_score, task_cost, num_attempts)
        """
        with submission_path.open() as f:
            task_submission = json.load(f)

        task_score = 0
        num_pairs = len(self.solutions[task_id]['test'])
        task_cost = 0.0
        num_attempts = 0
        pixel_similarities = []

        for enum_pair_index, pair_attempts in enumerate(task_submission):
            # Try to extract pair_index from the attempt key
            pair_index = None
            
            # First, check if pair_index is directly in the attempt data
            for attempt_key in pair_attempts:
                if (pair_attempts[attempt_key] is not None and 
                    isinstance(pair_attempts[attempt_key], dict) and 
                    'pair_index' in pair_attempts[attempt_key]['metadata']):
                    pair_index = pair_attempts[attempt_key]['metadata']['pair_index']
                    if 0 <= pair_index < num_pairs:  # Validate the index
                        break
            
            # If not found in data, fall back to enumeration index
            if pair_index is None:
                pair_index = enum_pair_index
                
            # Skip if the pair_index is out of bounds
            if pair_index >= num_pairs:
                self.print_log(f"    Warning: Invalid pair_index {pair_index} for {task_id}, skipping")
                continue
            
            # Count all attempts in this pair, regardless of whether we process them all
            num_attempts += len(pair_attempts)
            
            # Calculate costs for all attempts upfront
            for attempt_key in pair_attempts:
                if pair_attempts[attempt_key] is None:
                    continue
                
                if 'metadata' in pair_attempts[attempt_key] and 'cost' in pair_attempts[attempt_key]['metadata']:
                    attempt_cost = pair_attempts[attempt_key]['metadata']['cost'].get('total_cost', 0.0)
                    task_cost += attempt_cost
            
            pair_correct = False
            best_similarity = 0.0
            
            for attempt_key in pair_attempts:
                attempt_data = pair_attempts[attempt_key]

                if attempt_data is None:
                    self.print_log(f"    No prediction for {task_id}, pair {pair_index}, attempt {attempt_key}")
                    continue

                if attempt_data == []:
                    self.print_log(f"    No prediction for {task_id}, pair {pair_index}, attempt {attempt_key}")
                    continue

                # Check for exact match
                if attempt_data['answer'] == self.solutions[task_id]['test'][pair_index]['output']:
                    pair_correct = True
                    best_similarity = 1.0
                    break
                
                # Calculate pixel similarity if not an exact match
                similarity = self.calculate_pixel_similarity(
                    attempt_data['answer'], 
                    self.solutions[task_id]['test'][pair_index]['output']
                )
                best_similarity = max(best_similarity, similarity)

            if pair_correct:
                task_score += 1
            
            pixel_similarities.append(best_similarity)

        # Calculate average pixel similarity for this task
        avg_pixel_similarity = sum(pixel_similarities) / len(pixel_similarities) if pixel_similarities else 0.0

        scoring_result = {
            "score": task_score / num_pairs,
            "cost": task_cost,
            "attempts": num_attempts,
            "pixel_similarity": avg_pixel_similarity
        }

        return scoring_result

    def score_submission(self) -> Tuple[float, int]:
        """
        Read a submission from file, score it, then return the score
        """
        # self.print_log(f"Scoring {self.submission_dir}\n")

        total_score = 0
        total_tasks = 0
        total_pixel_similarity = 0.0
        task_results = {}

        for submission_file in self.submission_dir.glob('*.json'):
            if submission_file.name == 'results.json':
                continue

            task_id = submission_file.stem
            scoring_result = self.score_task(task_id, submission_file)
            
            total_score += scoring_result["score"]
            total_tasks += 1
            self.total_cost += scoring_result["cost"]
            self.total_attempts += scoring_result["attempts"]
            total_pixel_similarity += scoring_result["pixel_similarity"]
            
            task_results[task_id] = {
                "score": scoring_result['score'],
                "cost": scoring_result['cost'],
                "attempts": scoring_result['attempts'],
                "pixel_similarity": scoring_result['pixel_similarity']
            }

            self.print_log(f"    Task {task_id} score: {scoring_result['score']:.2f}, cost: ${scoring_result['cost']:.4f}, attempts: {scoring_result['attempts']}, pixel similarity: {scoring_result['pixel_similarity']:.4f}")

        # Calculate average costs and similarity
        avg_cost_per_task = self.total_cost / total_tasks if total_tasks > 0 else 0
        avg_cost_per_attempt = self.total_cost / self.total_attempts if self.total_attempts > 0 else 0
        avg_pixel_similarity = total_pixel_similarity / total_tasks if total_tasks > 0 else 0

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
                    "avg_pixel_similarity": avg_pixel_similarity,
                    "task_results": task_results
                }, f)

        # Calculate and print percentage score
        percentage_score = (total_score / total_tasks * 100) if total_tasks > 0 else 0
        print(f"\nFinal Score: {percentage_score:.2f}% ({total_score:.2f}/{total_tasks})")
        print(f"Average Pixel Similarity: {avg_pixel_similarity:.4f}")
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