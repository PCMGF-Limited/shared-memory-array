from multiprocessing.shared_memory import SharedMemory

import numpy as np
from multiprocessing.managers import SharedMemoryManager

from numpy._typing import DTypeLike, _ShapeLike  # noqa

from typing import NamedTuple


class SharedArray(NamedTuple):
    """
    A pointer to a numpy.ndarray in shared memory
    For details, see https://docs.python.org/3/library/multiprocessing.shared_memory.html

    Example that sums a matrix by summing each row in parallel:
    - create a SharedMemoryManager, otherwise the shared memory would not be freed
    - create a shared copy of the array SharedArray.copy(manager, x)
    - pool workers unpack the shared array with the as_array() method
    - use pool.imap to get a generator for tdqm

    ```
    def sum_row(sai):
      sa, i = sai
      x = sa.as_array()
      return x[i].sum()

    def sum_by_row():
        x = np.ones([1000, 1000])
        with SharedMemoryManager() as manager:
            snp = SharedArray.copy(manager, x)
            with Pool() as pool:
                result_generator = pool.imap(sum_row, ((snp, i) for i in range(1000)))
                result_generator = tqdm(result_generator, total=1000, position=0, leave=True)
                return sum(result_generator)
    ```

    For automatic cleanup, use the managed() context manager:
    ```
    with SharedMemoryManager() as manager:
        with SharedArray.allocate(manager, (1000, 1000), 'float64').managed() as sa:
            x = sa.as_array()
            x.fill(42.0)
            # Automatically closed on exit
    ```
    """
    shm: SharedMemory
    shape: _ShapeLike
    dtype: DTypeLike
    owner: bool = True

    @property
    def name(self) -> str:
        """Get the shared memory block name."""
        return self.shm.name

    def as_array(self) -> np.ndarray:
        """
        Reconstruct the numpy array from shared memory buffer.

        Returns:
            np.ndarray: View of the shared memory as a numpy array
        """
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self) -> None:
        """
        Close the shared memory handle and unlink if owner.

        Closes *this* handle. Unlinks if owner.
        This is idempotent - calling multiple times is safe.
        """
        # try:
        self.shm.close()
        # except Exception:  TODO include specific exception here
        # pass

        if self.owner:
            self.unlink()

    def unlink(self) -> None:
        """
        Unlink the shared memory block.

        unlink() should be called only once per shared memory block.
        This is idempotent - calling multiple times is safe.
        """
        try:
            self.shm.unlink()
        except FileNotFoundError:
            # Idempotent cleanup: already unlinked elsewhere.
            pass

    def managed(self) -> "ManagedSharedArray":
        """
        Return a context manager wrapper for automatic cleanup.

        Returns:
            ManagedSharedArray: Context manager wrapper

        Example:
            with sa.managed() as managed_sa:
                arr = managed_sa.as_array()
                # Automatically closed on exit
        """
        return ManagedSharedArray(self)


    @staticmethod
    def _validate_shape(shape: _ShapeLike) -> tuple:
        """
        Validate and normalize shape parameter.

        Args:
            shape: Array shape specification

        Returns:
            tuple: Normalized shape as tuple of integers

        Raises:
            ValueError: If shape contains invalid values
            TypeError: If shape cannot be converted to tuple of integers
        """
        # Reject None and other clearly invalid types
        if shape is None or isinstance(shape, str):
            raise TypeError(f"Shape must be iterable or integer, got {type(shape)}")

        try:
            # Try to convert to tuple
            if hasattr(shape, '__iter__'):
                shape_tuple = tuple(shape)
            else:
                # Single integer case
                shape_tuple = (shape,)
        except TypeError:
            raise TypeError(f"Shape must be iterable or integer, got {type(shape)}")

        # Validate all dimensions are integers
        if not all(isinstance(d, (int, np.integer)) for d in shape_tuple):
            raise TypeError(f"Shape dimensions must be integers, got {shape_tuple}")

        # Validate all dimensions are non-negative
        if any(d < 0 for d in shape_tuple):
            raise ValueError(f"Shape dimensions must be non-negative, got {shape_tuple}")

        return shape_tuple

    @staticmethod
    def _validate_dtype(dtype: DTypeLike) -> np.dtype:
        """
        Validate dtype parameter.

        Args:
            dtype: NumPy dtype specification

        Returns:
            np.dtype: Normalized dtype object

        Raises:
            TypeError: If dtype is invalid
        """
        if dtype is None:
            raise TypeError("Invalid dtype specification: dtype cannot be None")

        try:
            dt = np.dtype(dtype)
        except TypeError as e:
            raise TypeError(f"Invalid dtype specification: {dtype}") from e

        return dt

    @staticmethod
    def _calculate_buffer_size(shape: tuple, dtype: np.dtype) -> int:
        """
        Calculate required buffer size for array.

        Args:
            shape: Array shape tuple
            dtype: NumPy dtype object

        Returns:
            int: Required buffer size in bytes
        """
        n_elements = int(np.prod(shape))
        itemsize = dtype.itemsize
        return n_elements * itemsize

    @staticmethod
    def _validate_buffer_size(shm: SharedMemory, shape: tuple, dtype: np.dtype):
        """
        Validate that shared memory buffer is correct size.

        Args:
            shm: SharedMemory object
            shape: Expected array shape
            dtype: Expected array dtype

        Raises:
            ValueError: If buffer size doesn't match requirements
        """
        required_size = SharedArray._calculate_buffer_size(shape, dtype)
        if shm.size < required_size:
            raise ValueError(
                f"Shared memory buffer too small: need {required_size} bytes "
                f"for shape {shape} and dtype {dtype}, but buffer is {shm.size} bytes"
            )

    @staticmethod
    def allocate_like(manager: SharedMemoryManager, array: np.ndarray):
        """
        Allocate shared memory matching an existing array's shape and dtype.

        Args:
            manager: SharedMemoryManager instance
            array: Template array

        Returns:
            SharedArray: New shared array with same shape and dtype

        Raises:
            ValueError: If allocation fails or array properties are invalid
        """
        shape = SharedArray._validate_shape(array.shape)
        dtype = SharedArray._validate_dtype(array.dtype)
        required_size = SharedArray._calculate_buffer_size(shape, dtype)

        # Shared memory doesn't support 0-byte allocations
        # Allocate minimum of 1 byte for zero-size arrays
        alloc_size = max(required_size, 1)

        try:
            shm = manager.SharedMemory(alloc_size)
        except Exception as e:
            raise ValueError(
                f"Failed to allocate {required_size} bytes of shared memory "
                f"for shape {shape} and dtype {dtype}"
            ) from e

        # Validate allocation succeeded with correct size
        # For zero-size arrays, we allocated 1 byte
        if required_size > 0:
            SharedArray._validate_buffer_size(shm, shape, dtype)

        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=True)

    @staticmethod
    def allocate(manager: SharedMemoryManager, shape: _ShapeLike, dtype: DTypeLike) -> "SharedArray":
        """
        Allocate shared memory for a new array.

        Args:
            manager: SharedMemoryManager instance
            shape: Array shape
            dtype: Array data type

        Returns:
            SharedArray: New shared array

        Raises:
            ValueError: If allocation fails or parameters are invalid
            TypeError: If shape or dtype are invalid types
        """
        shape = SharedArray._validate_shape(shape)
        dtype = SharedArray._validate_dtype(dtype)
        required_size = SharedArray._calculate_buffer_size(shape, dtype)

        # Shared memory doesn't support 0-byte allocations
        # Allocate minimum of 1 byte for zero-size arrays
        alloc_size = max(required_size, 1)

        try:
            shm = manager.SharedMemory(alloc_size)
        except Exception as e:
            raise ValueError(
                f"Failed to allocate {required_size} bytes of shared memory "
                f"for shape {shape} and dtype {dtype}"
            ) from e

        # Validate allocation succeeded with correct size
        # For zero-size arrays, we allocated 1 byte so we skip
        if required_size > 0:
            SharedArray._validate_buffer_size(shm, shape, dtype)

        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=True)

    @staticmethod
    def copy(manager: SharedMemoryManager, array: np.ndarray):
        """
        Copy an array into shared memory.

        Args:
            manager: SharedMemoryManager instance
            array: Source array to copy

        Returns:
            SharedArray: New shared array containing copy of data

        Raises:
            ValueError: If allocation or copy fails
        """
        sa = SharedArray.allocate_like(manager, array)

        # No need to copy data for zero-size arrays
        if array.size == 0:
            return sa

        dest = sa.as_array()

        # Handle non-contiguous arrays properly
        try:
            if array.flags.c_contiguous or array.flags.f_contiguous:
                dest[:] = array
            else:
                # For non-contiguous arrays, use np.copyto which handles it correctly
                np.copyto(dest, array)
        except Exception as e:
            raise ValueError(
                f"Failed to copy array data to shared memory. "
                f"Array shape: {array.shape}, dtype: {array.dtype}"
            ) from e

        return sa

    @staticmethod
    def attach(name, shape, dtype) -> "SharedArray":
        """
        Attach to existing shared memory by name.

        Args:
            name: Shared memory block name
            shape: Array shape
            dtype: Array data type

        Returns:
            SharedArray: SharedArray attached to existing memory

        Raises:
            ValueError: If buffer size doesn't match requirements
        """
        shape = SharedArray._validate_shape(shape)
        dtype = SharedArray._validate_dtype(dtype)
        shm = SharedMemory(name=name)

        # Only validate buffer size for non-zero arrays
        required_size = SharedArray._calculate_buffer_size(shape, dtype)
        if required_size > 0:
            SharedArray._validate_buffer_size(shm, shape, dtype)

        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=False)

    @staticmethod
    def allocate_like_unsafe(manager: SharedMemoryManager, array: np.ndarray):
        """Allocate without validation (unsafe, for performance)."""
        shm = manager.SharedMemory(array.nbytes)
        return SharedArray(shm=shm, shape=array.shape, dtype=array.dtype, owner=True)

    @staticmethod
    def allocate_unsafe(manager: SharedMemoryManager, shape: _ShapeLike, dtype: DTypeLike, nbytes=None):
        """Allocate without validation (unsafe, for performance)."""
        nbytes = nbytes or np.zeros(1, dtype=dtype).nbytes
        # nbytes = nbytes or np.dtype(dtype).itemsize
        shm = manager.SharedMemory(np.prod(shape) * nbytes)
        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=True)

    @staticmethod
    def copy_unsafe(manager: SharedMemoryManager, array: np.ndarray):
        """Copy without validation (unsafe, for performance)."""
        sa = SharedArray.allocate_like_unsafe(manager, array)
        a = sa.as_array()
        a[:] = array[:]
        return sa

    @staticmethod
    def attach_unsafe(name, shape, dtype):
        """Attach without validation (unsafe, for performance)."""
        shm = SharedMemory(name=name)
        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=False)


class ManagedSharedArray:
    """
    Context manager wrapper for SharedArray that provides automatic cleanup.

    This class wraps a SharedArray and calls close() automatically on exit.
    It delegates all attribute access to the underlying SharedArray.

    Example:
        with SharedArray.allocate(manager, (100,), 'float64').managed() as sa:
            arr = sa.as_array()
            arr.fill(42.0)
            # Automatically closed on exit
    """

    __slots__ = ('_shared_array', '_closed')

    def __init__(self, shared_array: SharedArray):
        """
        Initialize the managed wrapper.

        Args:
            shared_array: SharedArray instance to manage
        """
        object.__setattr__(self, '_shared_array', shared_array)
        object.__setattr__(self, '_closed', False)

    def __enter__(self) -> SharedArray:
        """
        Context manager entry.

        Returns:
            SharedArray: The wrapped SharedArray instance
        """
        if self._closed:
            raise ValueError("Cannot enter context: ManagedSharedArray is already closed")
        return self._shared_array

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically closes the shared memory."""
        if not self._closed:
            self._shared_array.close()
            object.__setattr__(self, '_closed', True)
        return False  # Don't suppress exceptions

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped SharedArray."""
        return getattr(self._shared_array, name)

    def __repr__(self):
        """String representation."""
        return f"ManagedSharedArray({self._shared_array!r})"