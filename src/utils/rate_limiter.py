import asyncio
import time
from typing import Optional

class AsyncTokenBucketLimiter:
    """
    An asynchronous token bucket rate limiter using asyncio.

    Best Practices Employed:
    - asyncio.Lock for atomic state updates in concurrent environments.
    - time.monotonic() for reliable interval timing, immune to clock changes.
    - Async context manager (__aenter__, __aexit__) for clean usage.
    - Efficient waiting with asyncio.sleep, releasing the lock while waiting.
    - Refill-on-demand logic within the lock for simplicity and safety.
    - Floating-point tokens for accuracy.
    """
    def __init__(self, rate: float, capacity: float):
        """
        Initializes the limiter.

        Args:
            rate: Tokens added per second (e.g., 5 for 5 RPS).
            capacity: Maximum tokens the bucket can hold (burst capacity).
        """
        if not (rate > 0 and capacity >= 0):
            raise ValueError("Rate must be positive and capacity non-negative")
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens = float(capacity) # Start full
        # Use monotonic clock for interval calculations
        self._last_refill_time = time.monotonic()
        # Lock to protect access to _tokens and _last_refill_time
        self._lock = asyncio.Lock()

    def _refill(self):
        """Calculates and adds tokens based on elapsed time. MUST be called inside the lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        if elapsed > 0:
            # In Python 3.8+, time.monotonic() resolution is nanoseconds, so elapsed can be tiny
            # Only refill if meaningful time has passed to avoid float precision issues
            # and unnecessary lock contention if called extremely rapidly.
            # A small epsilon check might be overly complex; just checking > 0 is usually sufficient.
            new_tokens = elapsed * self._rate
            self._tokens = min(self._capacity, self._tokens + new_tokens)
            self._last_refill_time = now # Always update last check time

    async def acquire(self, tokens_needed: int = 1) -> None:
        """
        Acquires the specified number of tokens, waiting if necessary.

        Args:
            tokens_needed: The number of tokens required for the operation.
                           Defaults to 1.

        Raises:
            ValueError: If tokens_needed is not positive or exceeds capacity.
        """
        if not isinstance(tokens_needed, int) or tokens_needed <= 0:
            raise ValueError("tokens_needed must be a positive integer")
        if tokens_needed > self._capacity:
            # Optimization: Don't even try if request exceeds capacity
            raise ValueError(f"Requested tokens ({tokens_needed}) exceeds bucket capacity ({self._capacity}) - acquisition impossible.")

        while True:
            async with self._lock: # Acquire lock for atomic check/update
                self._refill() # Refill based on time elapsed since last check

                if self._tokens >= tokens_needed:
                    # Sufficient tokens available
                    self._tokens -= tokens_needed
                    # print(f"Acquired {tokens_needed}. Remaining: {self._tokens:.2f}") # DEBUG
                    return # Success! Exit loop and context manager

                # Insufficient tokens, calculate wait time
                needed = tokens_needed - self._tokens
                wait_time = needed / self._rate
                # print(f"Insufficient tokens ({self._tokens:.2f}/{tokens_needed}). Waiting {wait_time:.3f}s") # DEBUG

            # ---- Lock is released before sleeping ----
            await asyncio.sleep(wait_time)
            # Loop continues, will re-acquire lock, refill and try again

    async def __aenter__(self):
        """Async context manager entry point (acquires 1 token)."""
        await self.acquire(1) # Default to acquiring 1 token
        # __aenter__ must return something usable with 'as', self is common
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point (no action needed here)."""
        pass # No token release necessary for this type of limiter

    def get_current_tokens(self) -> float:
         """Returns the approximate current number of tokens (primarily for inspection/debugging).
         Refills based on elapsed time since last check before returning.
         Uses lock for consistency.
         """
         # Although reading might seem safe, we need the lock to ensure
         # we call _refill first for an up-to-date value.
         # This makes it consistent with the acquire logic.
         # If this becomes a performance issue (high contention just for reading),
         # consider alternatives, but correctness is usually preferred.
         async def _get_tokens():
             async with self._lock:
                 self._refill()
                 return self._tokens
         # If called from sync code, need to run async logic
         try:
             loop = asyncio.get_running_loop()
             # If in async context, run directly (avoids blocking event loop)
             return asyncio.run_coroutine_threadsafe(_get_tokens(), loop).result()
         except RuntimeError:
             # If not in async context, run in a new event loop (use sparingly)
             # print("Warning: get_current_tokens called from sync code, may block.")
             return asyncio.run(_get_tokens()) 