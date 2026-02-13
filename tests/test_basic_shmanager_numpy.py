import pytest
import numpy as np
from multiprocessing.managers import SharedMemoryManager

from shared_memory_array import SharedMemoryArray


class TestValidateShape:
    """Tests for _validate_shape() method."""

    def test_valid_tuple_shape(self):
        """Test valid tuple shapes."""
        assert SharedMemoryArray._validate_shape((10, 20)) == (10, 20)
        assert SharedMemoryArray._validate_shape((5,)) == (5,)
        assert SharedMemoryArray._validate_shape((1, 2, 3, 4)) == (1, 2, 3, 4)

    def test_valid_list_shape(self):
        """Test valid list shapes."""
        assert SharedMemoryArray._validate_shape([10, 20]) == (10, 20)
        assert SharedMemoryArray._validate_shape([100]) == (100,)

    def test_valid_scalar_shape(self):
        """Test valid scalar shape."""
        assert SharedMemoryArray._validate_shape(10) == (10,)

    def test_zero_dimension(self):
        """Test that zero dimensions are allowed."""
        assert SharedMemoryArray._validate_shape((0, 10)) == (0, 10)
        assert SharedMemoryArray._validate_shape((10, 0)) == (10, 0)

    def test_numpy_integer_shape(self):
        """Test shape with numpy integer types."""
        assert SharedMemoryArray._validate_shape((np.int64(10), np.int32(20))) == (10, 20)

    def test_negative_dimension_raises(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SharedMemoryArray._validate_shape((10, -5))
        with pytest.raises(ValueError, match="non-negative"):
            SharedMemoryArray._validate_shape((-1,))

    def test_non_integer_dimension_raises(self):
        """Test that non-integer dimensions raise TypeError."""
        with pytest.raises(TypeError, match="must be integers"):
            SharedMemoryArray._validate_shape((10.5, 20))
        with pytest.raises(TypeError, match="must be integers"):
            SharedMemoryArray._validate_shape(("10", 20))

    def test_non_iterable_non_integer_raises(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="must be iterable or integer"):
            SharedMemoryArray._validate_shape(None)
        with pytest.raises(TypeError, match="must be iterable or integer"):
            SharedMemoryArray._validate_shape("invalid")


class TestValidateDtype:
    """Tests for _validate_dtype() method."""

    def test_valid_string_dtype(self):
        """Test valid string dtype specifications."""
        assert SharedMemoryArray._validate_dtype('float64') == np.float64
        assert SharedMemoryArray._validate_dtype('int32') == np.int32
        assert SharedMemoryArray._validate_dtype('uint8') == np.uint8

    def test_valid_numpy_dtype(self):
        """Test valid numpy dtype objects."""
        assert SharedMemoryArray._validate_dtype(np.float32) == np.float32
        assert SharedMemoryArray._validate_dtype(np.int64) == np.int64

    def test_valid_python_type(self):
        """Test valid Python type specifications."""
        assert SharedMemoryArray._validate_dtype(float) == np.float64
        assert SharedMemoryArray._validate_dtype(int) == np.int64

    def test_invalid_dtype_raises(self):
        """Test that invalid dtypes raise TypeError."""
        with pytest.raises(TypeError, match="Invalid dtype"):
            SharedMemoryArray._validate_dtype("not_a_dtype")
        with pytest.raises(TypeError, match="Invalid dtype"):
            SharedMemoryArray._validate_dtype(None)


class TestCalculateBufferSize:
    """Tests for _calculate_buffer_size() method."""

    def test_simple_1d_array(self):
        """Test buffer size for 1D arrays."""
        shape = (100,)
        dtype = np.dtype(np.float64)
        expected = 100 * 8  # 100 elements * 8 bytes
        assert SharedMemoryArray._calculate_buffer_size(shape, dtype) == expected

    def test_simple_2d_array(self):
        """Test buffer size for 2D arrays."""
        shape = (10, 20)
        dtype = np.dtype(np.int32)
        expected = 10 * 20 * 4  # 200 elements * 4 bytes
        assert SharedMemoryArray._calculate_buffer_size(shape, dtype) == expected

    def test_multidimensional_array(self):
        """Test buffer size for multidimensional arrays."""
        shape = (5, 10, 15, 20)
        dtype = np.dtype(np.uint8)
        expected = 5 * 10 * 15 * 20 * 1  # 15000 elements * 1 byte
        assert SharedMemoryArray._calculate_buffer_size(shape, dtype) == expected

    def test_zero_size_array(self):
        """Test buffer size for zero-size arrays."""
        shape = (0, 10)
        dtype = np.dtype(np.float32)
        expected = 0
        assert SharedMemoryArray._calculate_buffer_size(shape, dtype) == expected

    def test_different_dtypes(self):
        """Test buffer size calculation with various dtypes."""
        shape = (10,)
        assert SharedMemoryArray._calculate_buffer_size(shape, np.dtype(np.float64)) == 80
        assert SharedMemoryArray._calculate_buffer_size(shape, np.dtype(np.float32)) == 40
        assert SharedMemoryArray._calculate_buffer_size(shape, np.dtype(np.int16)) == 20
        assert SharedMemoryArray._calculate_buffer_size(shape, np.dtype(bool)) == 10


class TestValidateBufferSize:
    """Tests for _validate_buffer_size() method."""

    def test_exact_size_passes(self):
        """Test that exact buffer size passes validation."""
        with SharedMemoryManager() as manager:
            shm = manager.SharedMemory(800)  # 100 * 8 bytes
            shape = (100,)
            dtype = np.dtype(np.float64)
            # Should not raise
            SharedMemoryArray._validate_buffer_size(shm, shape, dtype)

    def test_larger_size_passes(self):
        """Test that larger buffer size passes validation."""
        with SharedMemoryManager() as manager:
            shm = manager.SharedMemory(1000)  # More than needed
            shape = (100,)
            dtype = np.dtype(np.float64)
            # Should not raise
            SharedMemoryArray._validate_buffer_size(shm, shape, dtype)

    def test_smaller_size_raises(self):
        """Test that smaller buffer size raises ValueError."""
        with SharedMemoryManager() as manager:
            shm = manager.SharedMemory(10000)  # this is likely going to be the page size (e.g. 16kb)
            shape = (10000,)
            dtype = np.float64  # Needs 800 bytes

            with pytest.raises(ValueError, match="buffer too small"):
                SharedMemoryArray._validate_buffer_size(shm, shape, dtype)


class TestAllocate:
    """Tests for allocate() method."""

    def test_allocate_1d_float(self):
        """Test allocating 1D float array."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(100,), dtype=np.float64)
            assert sa.shape == (100,)
            assert sa.dtype == np.dtype(np.float64)
            arr = sa.as_array()
            assert arr.shape == (100,)
            assert arr.dtype == np.dtype(np.float64)

    def test_allocate_2d_int(self):
        """Test allocating 2D int array."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10, 20), dtype=np.int32)
            assert sa.shape == (10, 20)
            assert sa.dtype == np.dtype(np.int32)
            arr = sa.as_array()
            assert arr.shape == (10, 20)
            assert arr.dtype == np.dtype(np.int32)

    def test_allocate_multidimensional(self):
        """Test allocating multidimensional array."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(2, 3, 4, 5), dtype=np.uint8)
            assert sa.shape == (2, 3, 4, 5)
            arr = sa.as_array()
            assert arr.shape == (2, 3, 4, 5)

    def test_allocate_with_list_shape(self):
        """Test allocating with list shape."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=[5, 10], dtype=np.float32)
            assert sa.shape == (5, 10)

    def test_allocate_invalid_shape_raises(self):
        """Test that invalid shape raises appropriate error."""
        with SharedMemoryManager() as manager:
            with pytest.raises(ValueError, match="non-negative"):
                SharedMemoryArray.allocate(manager, shape=(-10, 20), dtype=np.float64)

            with pytest.raises(TypeError, match="must be integers"):
                SharedMemoryArray.allocate(manager, shape=(10.5, 20), dtype=np.float64)

    def test_allocate_invalid_dtype_raises(self):
        """Test that invalid dtype raises appropriate error."""
        with SharedMemoryManager() as manager:
            with pytest.raises(TypeError, match="Invalid dtype"):
                SharedMemoryArray.allocate(manager, shape=(10, 20), dtype='invalid_dtype')

    def test_allocate_zero_size_array(self):
        """Test allocating zero-size array."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(0, 10), dtype=np.float64)
            assert sa.shape == (0, 10)
            arr = sa.as_array()
            assert arr.size == 0


class TestAllocateLike:
    """Tests for allocate_like() method."""

    def test_allocate_like_1d_array(self):
        """Test allocate_like with 1D array."""
        template = np.zeros(100, dtype=np.float64)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate_like(manager, template)
            assert sa.shape == template.shape
            assert sa.dtype == template.dtype

    def test_allocate_like_2d_array(self):
        """Test allocate_like with 2D array."""
        template = np.ones((50, 30), dtype=np.int32)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate_like(manager, template)
            assert sa.shape == template.shape
            assert sa.dtype == template.dtype

    def test_allocate_like_various_dtypes(self):
        """Test allocate_like with various dtypes."""
        dtypes = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8]
        with SharedMemoryManager() as manager:
            for dtype in dtypes:
                template = np.zeros(10, dtype=dtype)
                sa = SharedMemoryArray.allocate_like(manager, template)
                assert sa.dtype == np.dtype(dtype)

    def test_allocate_like_empty_array(self):
        """Test allocate_like with empty array."""
        template = np.array([], dtype=np.float64)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate_like(manager, template)
            assert sa.shape == (0,)


class TestCopy:
    """Tests for copy() method."""

    def test_copy_contiguous_1d_array(self):
        """Test copying contiguous 1D array."""
        original = np.arange(100, dtype=np.float64)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            np.testing.assert_array_equal(copied, original)

    def test_copy_contiguous_2d_array(self):
        """Test copying contiguous 2D array."""
        original = np.arange(600).reshape(20, 30).astype(np.int32)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            np.testing.assert_array_equal(copied, original)

    def test_copy_non_contiguous_array(self):
        """Test copying non-contiguous array (sliced)."""
        original_full = np.arange(100, dtype=np.float64)
        original = original_full[::2]  # Every other element (non-contiguous)
        assert not original.flags.c_contiguous

        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            np.testing.assert_array_equal(copied, original)

    def test_copy_transposed_array(self):
        """Test copying transposed (non-contiguous) array."""
        original = np.arange(20).reshape(4, 5).T  # Transpose makes it non-contiguous
        assert not original.flags.c_contiguous

        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            np.testing.assert_array_equal(copied, original)

    def test_copy_sliced_2d_array(self):
        """Test copying sliced 2D array."""
        original_full = np.arange(100).reshape(10, 10)
        original = original_full[::2, ::3]  # Non-contiguous slice

        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            np.testing.assert_array_equal(copied, original)

    def test_copy_preserves_values(self):
        """Test that copy preserves exact values."""
        original = np.random.randn(50, 60)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            assert np.allclose(copied, original)
            assert copied.shape == original.shape
            assert copied.dtype == original.dtype

    def test_copy_empty_array(self):
        """Test copying empty array."""
        original = np.array([], dtype=np.float64)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            copied = sa.as_array()
            assert copied.size == 0


class TestAsArray:
    """Tests for as_array() method."""

    def test_as_array_returns_correct_shape(self):
        """Test that as_array returns correct shape."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10, 20), dtype=np.float64)
            arr = sa.as_array()
            assert arr.shape == (10, 20)

    def test_as_array_returns_correct_dtype(self):
        """Test that as_array returns correct dtype."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(100,), dtype=np.int32)
            arr = sa.as_array()
            assert arr.dtype == np.dtype(np.int32)

    def test_as_array_is_writable(self):
        """Test that as_array returns writable array."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.float64)
            arr = sa.as_array()
            arr[:] = 42.0
            assert np.all(arr == 42.0)

    def test_as_array_multiple_calls_same_data(self):
        """Test that multiple as_array calls access same underlying data."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.float64)
            arr1 = sa.as_array()
            arr1[:] = 99.0
            arr2 = sa.as_array()
            np.testing.assert_array_equal(arr1, arr2)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_allocate_fill_read(self):
        """Test complete workflow: allocate -> fill -> read."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(100, 100), dtype=np.float64)
            arr = sa.as_array()
            arr.fill(3.14)

            # Read back in "another process" (simulated by new as_array call)
            arr2 = sa.as_array()
            assert np.all(arr2 == 3.14)

    def test_copy_modify_independent(self):
        """Test that modifying shared array doesn't affect original."""
        original = np.ones((50, 50), dtype=np.float32)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.copy(manager, original)
            shared = sa.as_array()
            shared *= 2.0

            # Original should be unchanged
            assert np.all(original == 1.0)
            # Shared should be modified
            assert np.all(shared == 2.0)

    def test_large_array(self):
        """Test with large array to ensure memory handling."""
        shape = (1000, 1000)
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=shape, dtype=np.float64)
            arr = sa.as_array()
            assert arr.shape == shape
            arr[0, 0] = 42.0
            assert arr[0, 0] == 42.0

    def test_multiple_arrays_same_manager(self):
        """Test creating multiple arrays with same manager."""
        with SharedMemoryManager() as manager:
            sa1 = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.float64)
            sa2 = SharedMemoryArray.allocate(manager, shape=(20,), dtype=np.int32)
            sa3 = SharedMemoryArray.allocate(manager, shape=(5, 5), dtype=np.uint8)

            arr1 = sa1.as_array()
            arr2 = sa2.as_array()
            arr3 = sa3.as_array()

            arr1[:] = 1.0
            arr2[:] = 2
            arr3[:] = 3

            assert np.all(arr1 == 1.0)
            assert np.all(arr2 == 2)
            assert np.all(arr3 == 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
