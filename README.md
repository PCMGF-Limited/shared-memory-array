# shmanager

A lightweight Python library for efficiently sharing NumPy arrays across multiple processes using shared memory, eliminating expensive data copying in parallel workloads.

## The Problem

When processing large NumPy arrays in parallel, you typically face these trade-offs:

- **Sequential operations**: Simple but slow, defeats the purpose of parallelization
- **Reloading arrays per process**: Expensive in both memory and I/O time
- **Passing arrays to workers**: Each process gets a copy via pickle, multiplying memory usage
- **Multi-threading with GIL**: Limited by Python's Global Interpreter Lock, unpredictable performance
- **This library**: Share arrays in memory across processes with zero copying :+1:


## Installation

```bash
pip install shmanager
```

Or install from source:

```bash
git clone https://github.com/PCMGF-Limited/sharedmemarray.git
cd shmanager
pip install -e .
```


## Quick Start

### Basic Example: Parallel Row Summation

```python
import numpy as np
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from shmanager import SharedArray

def sum_row(args):
    """Sum a single row - runs in parallel process."""
    shared_array, row_idx = args
    arr = shared_array.as_array()
    return arr[row_idx].sum()

# Create large array
data = np.random.rand(1000, 1000)

# Share it across processes
with SharedMemoryManager() as manager:
    shared = SharedArray.copy(manager, data)
    
    with Pool() as pool:
        # Each worker accesses the SAME memory - no copying!
        results = pool.map(sum_row, [(shared, i) for i in range(1000)])
    
    print(f"Total sum: {sum(results)}")
```


### With Automatic Cleanup (Context Manager)

```python
with SharedMemoryManager() as manager:
    # Automatically closes/unlinks on exit
    with SharedArray.allocate(manager, shape=(10000, 10000), dtype='float64').managed() as shared:
        arr = shared.as_array()
        arr[:] = 42.0
        # Process array in parallel...
    # Memory automatically cleaned up here
```


### Attach from Multiple Processes

```python
# Process 1: Create and populate
with SharedMemoryManager() as manager:
    owner = SharedArray.allocate(manager, shape=(1000, 1000), dtype='float64')
    arr = owner.as_array()
    arr[:] = np.random.rand(1000, 1000)
    
    name = owner.name  # Share this name with other processes
    
    # Process 2: Attach by name
    client = SharedArray.attach(name=name, shape=(1000, 1000), dtype='float64')
    client_arr = client.as_array()
    # Both processes see the same data!
    
    client.close()  # Non-owner: closes handle but doesn't unlink
    owner.close()   # Owner: closes and unlinks memory
```


## Core API

### Creating Shared Arrays

```python
# Allocate new shared array
sa = SharedArray.allocate(manager, shape=(100, 200), dtype='float64')

# Allocate matching existing array
template = np.zeros((50, 50), dtype='int32')
sa = SharedArray.allocate_like(manager, template)

# Copy existing array into shared memory
data = np.random.rand(1000, 1000)
sa = SharedArray.copy(manager, data)
```


### Accessing Arrays

```python
# Get numpy array view (no copying!)
arr = sa.as_array()
arr[0, 0] = 42.0  # Modifications visible to all processes

# Access metadata
print(sa.name)   # Shared memory block name
print(sa.shape)  # Array shape
print(sa.dtype)  # Array dtype
print(sa.owner)  # Whether this instance owns the memory
```


### Cleanup

```python
# Manual cleanup
sa.close()  # Closes handle; unlinks if owner=True

# Automatic cleanup with context manager
with sa.managed() as managed_sa:
    arr = managed_sa.as_array()
    # Use array...
# Automatically closed here
```


### Attaching to Existing Memory

```python
# Attach to shared memory by name (e.g., from another process)
client = SharedArray.attach(
    name="psm_12345",
    shape=(100, 100),
    dtype='float64'
)
# owner=False by default (won't unlink on close)
```


## Safe vs Unsafe Methods

For performance-critical code where you control all inputs, use `_unsafe` variants that skip validation:

```python
# Safe (default): validates shape, dtype, buffer sizes
sa_safe = SharedArray.allocate(manager, shape=(100,), dtype='float64')

# Unsafe: skips validation for performance
sa_unsafe = SharedArray.allocate_unsafe(manager, shape=(100,), dtype='float64')

# Also available: copy_unsafe, allocate_like_unsafe, attach_unsafe
```

:warning: **Warning**: Unsafe methods can cause segfaults or data corruption if used incorrectly. Only use when you've validated inputs yourself.

## Features

- **Zero-copy sharing**: Multiple processes access the same memory
- **NumPy integration**: Returns standard numpy arrays
- **Context managers**: Automatic cleanup on exit
- **Type safety**: Full validation of shapes, dtypes, and buffer sizes
- **Non-contiguous arrays**: Handles slices and transposed arrays correctly
- **Ownership tracking**: Distinguishes between owners and clients
- **Performance mode**: Unsafe variants skip validation overhead


## Performance Comparison

```python
import time
import numpy as np
from multiprocessing import Pool
from shmanager import SharedArray

data = np.random.rand(10000, 10000)  # ~800MB array

# Traditional approach: pickle copies data to each worker
def traditional():
    with Pool() as pool:
        results = pool.map(process_chunk, [(data, i) for i in range(100)])
    # Total memory: 800MB × num_workers 😱

# With shmanager: zero copying
def with_shmanager():
    with SharedMemoryManager() as manager:
        shared = SharedArray.copy(manager, data)
        with Pool() as pool:
            results = pool.map(process_chunk_shared, [(shared, i) for i in range(100)])
    # Total memory: 800MB regardless of workers 🎉
```


## Complete Example: Image Processing Pipeline

```python
import numpy as np
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from shmanager import SharedArray
from tqdm import tqdm

def process_tile(args):
    """Process one tile of a large image."""
    shared_img, y_start, y_end = args
    img = shared_img.as_array()
    tile = img[y_start:y_end, :]
    # Apply expensive operation
    return tile.mean(), tile.std()

def parallel_image_analysis(image, n_tiles=10):
    """Analyze large image in parallel without copying."""
    tile_height = image.shape[0] // n_tiles
    
    with SharedMemoryManager() as manager:
        # Share image once
        shared = SharedArray.copy(manager, image)
        
        # Create tile boundaries
        tiles = [(shared, i*tile_height, (i+1)*tile_height) 
                 for i in range(n_tiles)]
        
        # Process in parallel
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_tile, tiles), 
                              total=n_tiles))
        
        return results

# Usage
large_image = np.random.rand(10000, 10000)
stats = parallel_image_analysis(large_image)
print(f"Processed {len(stats)} tiles")
```
## Notes

**Shared memory persists until explicitly unlinked!**

Always use context managers or call `.close()` to prevent memory leaks!

## Requirements

- Python 3.8+
- NumPy
- multiprocessing (standard library)


## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built on Python's `multiprocessing.shared_memory` module (Python 3.8+).

***




<img src="https://avatars.githubusercontent.com/u/75027313?s=32&v=4" style="height:32px;margin-right:0px; vertical-align: middle"/> PCMGF Limited