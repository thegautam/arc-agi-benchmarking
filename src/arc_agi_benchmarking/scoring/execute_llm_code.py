import argparse
import json
import re
import subprocess
import sys
import tempfile
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arc_agi_benchmarking.schemas import (
    ARCTask,
    BenchmarkedTaskResults,
    TestPairAttempts,
    Attempt,
)


def load_tasks(task_dir: Path) -> Dict[str, ARCTask]:
    tasks: Dict[str, ARCTask] = {}
    for solution_file in task_dir.glob("*.json"):
        task_id = solution_file.stem
        with solution_file.open() as f:
            tasks[task_id] = ARCTask.from_dict(json.load(f))
    return tasks


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_from_message(content: str) -> Optional[str]:
    # Try JSON object with a "code" field
    try:
        data = json.loads(content)
        if isinstance(data, dict) and isinstance(data.get("code"), str):
            return data["code"]
    except Exception:
        pass

    # Try fenced code blocks, return the longest block
    blocks = _CODE_BLOCK_RE.findall(content or "")
    if blocks:
        # Prefer blocks that define a transform-like function
        blocks_sorted = sorted(
            blocks,
            key=lambda b: ("def transform" in b) or ("def solve" in b) or ("def apply" in b),
            reverse=True,
        )
        return blocks_sorted[0]

    # Heuristic: if content contains 'def transform(' return the region around it
    if content and "def transform(" in content:
        return content

    return None


RUNNER_CODE = r"""
import sys, json, ast, io, contextlib, importlib

ALLOWED_MODULES = {
    'types',
    'collections',
    'collections.abc',
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError('Relative imports are not allowed')
    allowed = ALLOWED_MODULES
    base = name.split('.')[0]
    allowed_bases = {m.split('.')[0] for m in allowed}
    if name not in allowed and base not in allowed_bases:
        raise ImportError(f'Import of module "{name}" is not allowed')
    mod = importlib.import_module(name)
    if not fromlist:
        return importlib.import_module(base)
    return mod

SAFE_BUILTINS = {
    'range': range, 'len': len, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
    'enumerate': enumerate, 'map': map, 'filter': filter, 'list': list,
    'all': all, 'any': any, 'zip': zip,
    'set': set, 'sorted': sorted, 'tuple': tuple, 'int': int, 'float': float, 'bool': bool,
    'dict': dict, 'str': str, 'reversed': reversed, 'print': print,
    'isinstance': isinstance, 'issubclass': issubclass,
    '__import__': safe_import,
}


def _validate_imports(tree: ast.AST):
    allowed = ALLOWED_MODULES
    allowed_bases = {m.split('.')[0] for m in allowed}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                base = name.split('.')[0]
                if name not in allowed and base not in allowed_bases:
                    raise RuntimeError(f'Import of module "{name}" is not allowed')
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise RuntimeError('Relative imports are not allowed')
            mod = node.module or ''
            base = mod.split('.')[0] if mod else ''
            if mod not in allowed and base not in allowed_bases:
                raise RuntimeError(f'Import from "{mod}" is not allowed')

def safe_exec(code: str, g: dict):
    tree = ast.parse(code, mode='exec')
    _validate_imports(tree)
    g['__builtins__'] = SAFE_BUILTINS
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, '<llm>', 'exec'), g)


def discover_fn(g: dict):
    for name in ('transform', 'solve', 'apply', 'arc_transform'):
        fn = g.get(name)
        if callable(fn):
            return fn
    return None


def main():
    data = json.load(sys.stdin)
    code = data['code']
    input_grid = data.get('input')

    g: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        safe_exec(code, g)
    pre_stdout = buf.getvalue()

    fn = discover_fn(g)
    if not fn:
        print(json.dumps({'error': 'no_callable_found', 'stdout': pre_stdout}))
        return

    try:
        output = fn(input_grid)
    except Exception as e:
        print(json.dumps({'error': f'call_failed: {e}', 'stdout': pre_stdout}))
        return

    print(json.dumps({'output': output, 'stdout': pre_stdout, 'error': None}))


if __name__ == '__main__':
    main()
"""


def run_code_attempt(
    code: str,
    input_grid: List[List[int]],
    timeout: int,
) -> Tuple[Optional[List[List[int]]], str, Optional[str]]:
    """
    Execute code defining a transform-like function on a single input grid.
    Returns (output_grid, stdout, error)
    """
    payload = json.dumps({
        "code": code,
        "input": input_grid,
    })

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", RUNNER_CODE],
            input=payload.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", "timeout"

    if proc.returncode != 0:
        return None, proc.stdout.decode("utf-8", "ignore"), proc.stderr.decode("utf-8", "ignore")

    try:
        data = json.loads(proc.stdout.decode("utf-8", "ignore"))
    except Exception as e:
        return None, proc.stdout.decode("utf-8", "ignore"), f"bad_json:{e}"

    return (
        data.get("output"),
        data.get("stdout", ""),
        data.get("error"),
    )


def is_grid(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(r, list) and all(isinstance(v, int) for v in r) for r in x)


def verify_on_train(train_outs: List[Any], expected: List[List[List[int]]]) -> bool:
    if not isinstance(train_outs, list) or len(train_outs) != len(expected):
        return False
    for got, exp in zip(train_outs, expected):
        if not is_grid(got) or got != exp:
            return False
    return True


def _extract_bracketed_lists(text: str) -> List[str]:
    """Return substrings that are balanced square-bracketed lists from the text.
    We capture top-level bracketed segments by tracking nesting depth."""
    segs: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    segs.append(text[start:i + 1])
                    start = -1
    return segs


def parse_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """Parse the last 2D int grid from arbitrary stdout text.
    Tries JSON first, then ast.literal_eval on bracketed segments."""
    if not text:
        return None
    # Quick direct attempt on the whole text if it is just a JSON array
    t = text.strip()
    if t.startswith('[') and t.endswith(']'):
        for parser in (json.loads, ast.literal_eval):
            try:
                val = parser(t)
                if is_grid(val):
                    return val
            except Exception:
                pass
    # Extract balanced bracketed segments and try from the last one backwards
    segments = _extract_bracketed_lists(text)
    for seg in reversed(segments):
        for parser in (json.loads, ast.literal_eval):
            try:
                val = parser(seg)
                if is_grid(val):
                    return val
            except Exception:
                continue
    return None


def parse_assistant_output_grid(content: str) -> Optional[List[List[int]]]:
    """Parse a grid from assistant message JSON's 'output' field if present."""
    if not content:
        return None
    try:
        data = json.loads(content)
        out = data.get('output')
        if isinstance(out, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(out)
                    if is_grid(val):
                        return val
                except Exception:
                    continue
        elif is_grid(out):
            return out
    except Exception:
        # Not a JSON object; ignore
        return None
    return None


def build_executed_attempt(src_attempt: Attempt, answer: List[List[int]], pair_index: int, notes: Dict[str, Any]) -> Dict[str, Any]:
    # Convert pydantic Attempt to dict and update fields
    a = src_attempt.model_dump(mode="json")
    a["answer"] = answer
    # Ensure metadata exists and augment kwargs
    meta = a.get("metadata", {})
    kwargs = meta.get("kwargs", {}) or {}
    kwargs.update(notes)
    meta["kwargs"] = kwargs
    meta["pair_index"] = pair_index
    a["metadata"] = meta
    a["correct"] = None
    return a


def process_task(
    task_id: str,
    task: ARCTask,
    raw_json: Any,
    verify: bool,
    timeout: int,
    print_logs: bool = False,
) -> Any:
    # Normalize to BenchmarkedTaskResults for consistent handling
    btr = BenchmarkedTaskResults(test_pairs=raw_json)

    out_pairs: List[Any] = []

    for pair_index, pair_attempts in enumerate(btr):
        # Normalize attempts to a list
        attempts_list: List[Optional[Attempt]]
        if isinstance(pair_attempts.attempts, dict):
            attempts_list = [pair_attempts.attempts[k] for k in sorted(pair_attempts.attempts.keys())]
        else:
            attempts_list = list(pair_attempts.attempts)

        chosen_attempt_dict: Optional[Dict[str, Any]] = None

        for att in attempts_list:
            if att is None:
                continue
            # Extract code from latest message content
            try:
                content = att.metadata.choices[-1].message.content
            except Exception:
                content = ""
            code = extract_code_from_message(content or "")
            if not code:
                continue

            train_inputs = [p.input for p in task.train]
            expected_train = [p.output for p in task.train]
            test_input = task.test[pair_index].input

            # Compute train outputs by calling the simpler runner per input
            train_outs: List[Any] = []
            stdout_pieces: List[str] = []
            error: Optional[str] = None
            for grid in train_inputs:
                out_g, stdout_g, err_g = run_code_attempt(code, grid, timeout)
                stdout_pieces.append(stdout_g or "")
                if err_g and not error:
                    error = f"train_call_failed: {err_g}"
                train_outs.append(out_g)

            # Test output via single call
            test_out, stdout_test, err_test = run_code_attempt(code, test_input, timeout)
            stdout_text = ("\n".join(stdout_pieces + [stdout_test or ""]))[-2000:]
            if err_test and not error:
                error = f"test_call_failed: {err_test}"

            if error and print_logs:
                print(f"[task {task_id} pair {pair_index}] attempt error: {error}")

            ok = True
            if verify:
                ok = (train_outs is not None) and verify_on_train(train_outs, expected_train)

            if ok and is_grid(test_out):
                # Truncate code to avoid bloating JSON results
                code_snippet = (code or "")
                if len(code_snippet) > 8000:
                    code_snippet = code_snippet[:8000] + "\n...<truncated>"
                notes = {
                    "executed_by": "execute_llm_code.py",
                    "verified_on_train": bool(verify),
                    "stdout_snippet": (stdout_text or "")[-500:],
                    "extracted_code": code_snippet,
                }
                chosen_attempt_dict = build_executed_attempt(att, test_out, pair_index, notes)
                break

            # Fallback 1: parse grid from stdout if no valid test_out
            grid_from_stdout = parse_grid_from_text(stdout_text or "")
            if grid_from_stdout and is_grid(grid_from_stdout):
                code_snippet = (code or "")
                if len(code_snippet) > 8000:
                    code_snippet = code_snippet[:8000] + "\n...<truncated>"
                notes = {
                    "executed_by": "execute_llm_code.py",
                    "verified_on_train": False,
                    "stdout_parsed": True,
                    "stdout_snippet": (stdout_text or "")[-500:],
                    "extracted_code": code_snippet,
                }
                chosen_attempt_dict = build_executed_attempt(att, grid_from_stdout, pair_index, notes)
                break

            # Fallback 2: parse assistant JSON 'output' field
            grid_from_assistant = parse_assistant_output_grid(content or "")
            if grid_from_assistant and is_grid(grid_from_assistant):
                code_snippet = (code or "")
                if len(code_snippet) > 8000:
                    code_snippet = code_snippet[:8000] + "\n...<truncated>"
                notes = {
                    "executed_by": "execute_llm_code.py",
                    "verified_on_train": False,
                    "assistant_output_parsed": True,
                    "stdout_snippet": (stdout_text or "")[-500:],
                    "extracted_code": code_snippet,
                }
                chosen_attempt_dict = build_executed_attempt(att, grid_from_assistant, pair_index, notes)
                break

        # Fallback: if none chosen, create an empty attempt using first non-None attempt metadata
        if not chosen_attempt_dict:
            for att in attempts_list:
                if att is not None:
                    chosen_attempt_dict = build_executed_attempt(att, [], pair_index, {"executed_by": "execute_llm_code.py", "reason": "no_valid_attempt"})
                    break
            if not chosen_attempt_dict:
                # No attempts at all; synthesize a minimal structure
                chosen_attempt_dict = {
                    "answer": [],
                    "metadata": {"kwargs": {"executed_by": "execute_llm_code.py", "reason": "no_attempts"}, "pair_index": pair_index},
                    "correct": None,
                }

        # Store as TestPairAttempts-like object with single attempt per pair
        out_pairs.append({"attempts": [chosen_attempt_dict]})

    return out_pairs


def main():
    parser = argparse.ArgumentParser(description="Execute LLM-produced Python and generate submission outputs")
    parser.add_argument("--task_dir", required=True, type=str, help="Directory with ARC tasks (solutions)")
    parser.add_argument("--submission_dir", required=True, type=str, help="Directory with raw LLM submissions")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to write executed submissions")
    parser.add_argument("--timeout", type=int, default=10, help="Per-attempt execution timeout (seconds)")
    parser.add_argument("--no_verify", action="store_true", help="Do not verify on training examples")
    parser.add_argument("--print_logs", action="store_true")

    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    submission_dir = Path(args.submission_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(task_dir)

    for sub_file in submission_dir.glob("**/*.json"):
        if sub_file.name == "results.json":
            continue
        task_id = sub_file.stem
        if task_id not in tasks:
            if args.print_logs:
                print(f"Skipping {task_id}: not found in task_dir")
            continue
        with sub_file.open() as f:
            raw_json = json.load(f)

        executed_pairs = process_task(
            task_id,
            tasks[task_id],
            raw_json,
            verify=not args.no_verify,
            timeout=args.timeout,
            print_logs=args.print_logs,
        )

        out_path = out_dir / f"{task_id}.json"
        with out_path.open("w") as f:
            json.dump(executed_pairs, f, indent=2, default=str)
        if args.print_logs:
            print(f"Wrote executed submission: {out_path}")


if __name__ == "__main__":
    main()
