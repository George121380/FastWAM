"""Captures process start time on first import.

`scripts/train.py` imports this at the very top so we can later compute
"startup time" = wall seconds from process start to a chosen marker (e.g.
end of training step 10). Trainer reads `START` to emit the [perf]
startup_seconds line and embed it in profiling JSONL records.
"""

import time

START: float = time.perf_counter()
