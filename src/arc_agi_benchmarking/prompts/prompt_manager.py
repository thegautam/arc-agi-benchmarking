import json
import os
from arc_agi_benchmarking.schemas import ARCPair
from typing import List, Optional
from .scene_builder import build_scene_description, describe_grid

def _load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory
    """

    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, f"{prompt_name}.txt")
    
    with open(prompt_path, "r") as f:
        return f.read()

def convert_task_pairs_to_prompt(training_pairs: List[ARCPair], test_input: ARCPair, prompt_name: str = "agent_coding_prompt", prompt_feedback: Optional[str] = None) -> str:
    """
    Convert the training pairs to a prompt with scene descriptions.
    
    Args:
        training_pairs: List of training input/output pairs.
        test_input: The test input pair (only input is used for prompt construction).
        prompt_name: Name of the prompt template (without .txt) to load from this directory.
                      Defaults to "simple_coding_prompt".
    """
    prompt_template = _load_prompt(prompt_name)

    training_examples = ""
    for i, pair in enumerate(training_pairs):
        # Add scene description
        scene_desc = build_scene_description(pair.input, pair.output)
        
        training_examples += f"--Example {i}-- \n\n"
        training_examples += f"{scene_desc}\n\n"
        training_examples += "INPUT:\n\n"
        training_examples += json.dumps(pair.input) + "\n\n"
        training_examples += "OUTPUT:\n\n"
        training_examples += json.dumps(pair.output) + "\n\n"

    test_input_str = json.dumps(test_input.input)
    test_scene = describe_grid(test_input.input, "test input")

    # Build optional feedback section
    feedback_section = ""
    if prompt_feedback:
        feedback_section = (
            f"--Feedback from previous attempt--\n"
            f"{prompt_feedback}\n"
            f"--End of Feedback--\n\n"
        )

    return prompt_template.format(
        training_examples=training_examples,
        test_input=f"{test_scene}\n\n{test_input_str}",
        prompt_feedback_section=feedback_section,
    )