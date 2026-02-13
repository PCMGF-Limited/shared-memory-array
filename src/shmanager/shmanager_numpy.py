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
    """
    shm: SharedMemory
    shape: _ShapeLike
    dtype: DTypeLike
    owner: bool = True

    def as_array(self) -> np.ndarray:
        """
        Reconstruct the numpy array from shared memory buffer.

        Returns:
            np.ndarray: View of the shared memory as a numpy array
        """
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self) -> None:
        # Closes *this* handle. Unlinks if owner.
        self.shm.close()
        if self.owner:
            self.unlink()

    def unlink(self) -> None:
        # unlink() should be called only once per shared memory block.
        try:
            self.shm.unlink()
        except FileNotFoundError:
            # Idempotent cleanup: already unlinked elsewhere.
            pass


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
        try:
            shape_tuple = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
        except TypeError:
            raise TypeError(f"Shape must be iterable or integer, got {type(shape)}")

        if not all(isinstance(d, (int, np.integer)) for d in shape_tuple):
            raise TypeError(f"Shape dimensions must be integers, got {shape_tuple}")

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

        try:
            shm = manager.SharedMemory(required_size)
        except Exception as e:
            raise ValueError(
                f"Failed to allocate {required_size} bytes of shared memory "
                f"for shape {shape} and dtype {dtype}"
            ) from e

        # Validate allocation succeeded with correct size
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

        try:
            shm = manager.SharedMemory(required_size)
        except Exception as e:
            raise ValueError(
                f"Failed to allocate {required_size} bytes of shared memory "
                f"for shape {shape} and dtype {dtype}"
            ) from e

        # Validate allocation succeeded with correct size
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
    def attach(name, shape, dtype):
        shape = SharedArray._validate_shape(shape)
        dtype = SharedArray._validate_dtype(dtype)
        shm = SharedMemory(name=name)
        SharedArray._validate_buffer_size(shm, shape, dtype)
        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=False)

    @staticmethod
    def allocate_like_unsafe(manager: SharedMemoryManager, array: np.ndarray):
        shm = manager.SharedMemory(array.nbytes)
        return SharedArray(shm=shm, shape=array.shape, dtype=array.dtype, owner=True)

    @staticmethod
    def allocate_unsafe(manager: SharedMemoryManager, shape: _ShapeLike, dtype: DTypeLike, nbytes=None):
        nbytes = nbytes or np.zeros(1, dtype=dtype).nbytes
        # nbytes = nbytes or np.dtype(dtype).itemsize
        shm = manager.SharedMemory(np.prod(shape) * nbytes)
        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=True)

    @staticmethod
    def copy_unsafe(manager: SharedMemoryManager, array: np.ndarray):
        sa = SharedArray.allocate_like_unsafe(manager, array)
        a = sa.as_array()
        a[:] = array[:]
        return sa

    @staticmethod
    def attach_unsafe(name, shape, dtype):
        shm = SharedMemory(name=name)
        return SharedArray(shm=shm, shape=shape, dtype=dtype, owner=False)

