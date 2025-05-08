import time
import functools
import collections
import atexit
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Callable

# --- Global Storage & Control ---
_timing_data: List[Dict[str, Any]] = []
_counters = collections.Counter()
_output_dir = Path(os.environ.get("METRICS_OUTPUT_DIR", "metrics_output"))
METRICS_ENABLED = False  # Metrics are disabled by default


def set_metrics_enabled(enabled: bool):
    """Enable or disable metrics collection globally."""
    global METRICS_ENABLED
    METRICS_ENABLED = enabled
    if enabled:
        print("Metrics collection has been enabled.", flush=True)
    else:
        # This case might be less common if default is False, but good for explicit disabling
        print("Metrics collection has been disabled.", flush=True)

# --- Timing ---

def timeit(func: Callable) -> Callable:
    """Decorator to measure execution time of a function, if METRICS_ENABLED is True."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not METRICS_ENABLED:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        start_timestamp = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if METRICS_ENABLED: # Re-check in case it was disabled during func execution
                end_time = time.perf_counter()
                end_timestamp = time.time()
                duration = end_time - start_time
                _timing_data.append({
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "start_time_ns": int(start_time * 1e9), # Nanoseconds since epoch (perf_counter)
                    "end_time_ns": int(end_time * 1e9),     # Nanoseconds since epoch (perf_counter)
                    "duration_ms": duration * 1000,       # Milliseconds
                    "start_timestamp_utc": start_timestamp, # UTC timestamp
                    "end_timestamp_utc": end_timestamp,     # UTC timestamp
                    # Potentially add args/kwargs selectively if needed, but be careful with size/PII
                })
    return wrapper

def get_timing_data() -> List[Dict[str, Any]]:
    """Returns a copy of the collected timing data."""
    return list(_timing_data) # Return a copy

def dump_timing(filepath: str = "metrics_timing.csv"):
    """Saves the collected timing data to a CSV file, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        # print("Metrics are disabled, not dumping timing data.") # Optional: less verbose
        return
    if not _timing_data:
        print("No timing data collected.")
        return

    output_path = _output_dir / filepath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _timing_data[0].keys()
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_timing_data)
        print(f"Timing metrics saved to {output_path.resolve()}")
    except Exception as e:
        print(f"Error saving timing metrics to {output_path.resolve()}: {e}")

# --- Counting ---

def increment_counter(name: str, value: int = 1):
    """Increments a named counter, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        return
    _counters[name] += value

def get_counts() -> Dict[str, int]:
    """Returns a copy of the current counts."""
    return dict(_counters) # Return a copy

def dump_counts(filepath: str = "metrics_counts.json"):
    """Saves the counts to a JSON file, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        # print("Metrics are disabled, not dumping count data.") # Optional: less verbose
        return
    if not _counters:
        print("No count data collected.")
        return

    output_path = _output_dir / filepath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(get_counts(), f, indent=2, sort_keys=True)
        print(f"Count metrics saved to {output_path.resolve()}")
    except Exception as e:
        print(f"Error saving count metrics to {output_path.resolve()}: {e}")


# --- Automatic Dumping ---
def _dump_all():
    """Function called by atexit to dump all metrics, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        # print("\nMetrics collection is disabled. No metrics will be dumped.") # Less verbose if default disabled
        return
        
    print("\nDumping collected metrics...")
    dump_timing()
    dump_counts()
    print("Metrics dumping complete.")

atexit.register(_dump_all)

# --- Optional: Resetting (useful for testing) ---
def reset_metrics():
    """Clears all collected metrics and resets METRICS_ENABLED to its default (False)."""
    global _timing_data, _counters, METRICS_ENABLED
    _timing_data = []
    _counters = collections.Counter()
    METRICS_ENABLED = False # Reset to default disabled state
    print("Metrics reset and are disabled by default.") 