import os

# pyarrow (behind `datasets`) uses jemalloc by default, whose background thread can hold a
# lock while the DataLoader forks its workers; the forked worker then deadlocks (observed on
# 2026-10-04: the search hung for 1 h with the GPU idle). The system allocator avoids it and
# does not change any computation. Must be set before pyarrow is imported.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
