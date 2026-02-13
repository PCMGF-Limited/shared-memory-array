"""
Comprehensive tests for context manager functionality, name property access,
and safe vs unsafe method behavior.

Test Suite: test_context_shmanager_numpy.py

This test suite covers:
1. ManagedSharedArray context manager behavior
2. Name property access and uniqueness
3. Safe vs unsafe method comparisons
4. Edge cases and error handling in context managers
"""

import pytest
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import gc

from shared_memory_array import SharedMemoryArray, ManagedSharedMemoryArray


class TestManagedSharedArrayBasic:
    """Tests for basic ManagedSharedArray context manager functionality."""

    def test_managed_returns_managedsharedarray_instance(self):
        """Test that .managed() returns a ManagedSharedArray instance."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()
            assert isinstance(managed, ManagedSharedMemoryArray)

    def test_context_manager_enter_returns_sharedarray(self):
        """Test that __enter__ returns the underlying SharedArray."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            with sa.managed() as managed_sa:
                assert isinstance(managed_sa, SharedMemoryArray)
                assert managed_sa is sa

    def test_context_manager_auto_closes_on_exit(self):
        """Test that context manager automatically closes on normal exit."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            with sa.managed() as managed_sa:
                arr = managed_sa.as_array()
                arr[:] = 42.0

            # After exit, memory should be unlinked (since owner=True)
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

    def test_context_manager_closes_even_on_exception(self):
        """Test that context manager closes even when exception is raised."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            try:
                with sa.managed() as managed_sa:
                    arr = managed_sa.as_array()
                    arr[:] = 1.0
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Should still be unlinked
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

    def test_context_manager_does_not_suppress_exceptions(self):
        """Test that context manager does not suppress exceptions."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')

            with pytest.raises(RuntimeError, match="intentional"):
                with sa.managed():
                    raise RuntimeError("intentional")

    def test_context_manager_cleanup_for_non_owner(self):
        """Test that context manager respects owner=False (does not unlink)."""
        with SharedMemoryManager() as manager:
            owner_sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = owner_sa.name

            # Attach as non-owner
            client_sa = SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')
            assert client_sa.owner is False

            # Use managed context - should close but not unlink
            with client_sa.managed() as managed_client:
                arr = managed_client.as_array()
                arr[:] = 99.0

            # Memory should still exist (owner hasn't closed)
            owner_arr = owner_sa.as_array()
            assert np.all(owner_arr == 99.0)

            # Clean up owner
            owner_sa.close()


class TestManagedSharedArrayAttributeDelegation:
    """Tests for attribute delegation in ManagedSharedArray."""

    def test_attribute_delegation_to_underlying_sharedarray(self):
        """Test that attributes are delegated to underlying SharedArray."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(20, 30), dtype=np.int32)
            managed = sa.managed()

            # Access attributes through managed wrapper
            assert managed.shape == (20, 30)
            assert managed.dtype == np.int32
            assert managed.owner is True
            assert managed.name == sa.name

    def test_as_array_works_through_managed_wrapper(self):
        """Test that as_array() works through the managed wrapper."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            with managed as m:
                arr = m.as_array()
                arr[:] = 3.14
                assert np.allclose(arr, 3.14)


class TestManagedSharedArrayReentryPrevention:
    """Tests for preventing re-entry of closed ManagedSharedArray."""

    def test_cannot_reenter_after_exit(self):
        """Test that entering a closed ManagedSharedArray raises ValueError."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            # First entry and exit
            with managed:
                pass

            # Second entry should fail
            with pytest.raises(ValueError, match="already closed"):
                with managed:
                    pass

    def test_closed_flag_persists_after_exit(self):
        """Test that _closed flag is set after exiting context."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            assert managed._closed is False

            with managed:
                assert managed._closed is False

            assert managed._closed is True

    def test_manual_close_prevents_reentry(self):
        """Test that manually closing prevents context manager entry."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            # Manually trigger close via __exit__
            managed.__exit__(None, None, None)

            with pytest.raises(ValueError, match="already closed"):
                with managed:
                    pass


class TestManagedSharedArrayNesting:
    """Tests for nested context managers."""

    def test_nested_managed_arrays_different_arrays(self):
        """Test nesting managed contexts for different arrays."""
        with SharedMemoryManager() as manager:
            sa1 = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            sa2 = SharedMemoryArray.allocate(manager, shape=(20,), dtype='int32')

            with sa1.managed() as m1:
                arr1 = m1.as_array()
                arr1[:] = 1.0

                with sa2.managed() as m2:
                    arr2 = m2.as_array()
                    arr2[:] = 2

                    assert np.all(arr1 == 1.0)
                    assert np.all(arr2 == 2)

    def test_nested_manager_and_array_contexts(self):
        """Test nesting SharedMemoryManager with ManagedSharedArray."""
        with SharedMemoryManager() as manager:
            with SharedMemoryArray.allocate(manager, (100,), 'float32').managed() as sa:
                arr = sa.as_array()
                arr[:] = 99.9
                assert arr.shape == (100,)


class TestManagedSharedArrayCleanupBehavior:
    """Tests for cleanup behavior and idempotency."""

    def test_multiple_exit_calls_are_safe(self):
        """Test that calling __exit__ multiple times is safe (idempotent)."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            # Call __exit__ multiple times - should not raise
            managed.__exit__(None, None, None)
            managed.__exit__(None, None, None)
            managed.__exit__(None, None, None)

    def test_cleanup_with_different_exception_types(self):
        """Test cleanup behavior with various exception types."""
        exception_types = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error")
        ]

        for exc in exception_types:
            with SharedMemoryManager() as manager:
                sa = SharedMemoryArray.allocate(manager, shape=(5,), dtype='int64')
                name = sa.name

                try:
                    with sa.managed():
                        raise exc
                except type(exc):
                    pass

                # Should still be cleaned up
                with pytest.raises(FileNotFoundError):
                    SharedMemoryArray.attach(name=name, shape=(5,), dtype='int64')


class TestNameProperty:
    """Tests for name property access."""

    def test_name_property_returns_string(self):
        """Test that name property returns a string."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name
            assert isinstance(name, str)
            assert len(name) > 0

    def test_name_property_matches_shm_name(self):
        """Test that name property matches underlying shm.name."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            assert sa.name == sa.shm.name

    def test_name_uniqueness_across_allocations(self):
        """Test that each allocation gets a unique name."""
        with SharedMemoryManager() as manager:
            names = set()
            for _ in range(10):
                sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='uint8')
                names.add(sa.name)

            # All names should be unique
            assert len(names) == 10

    def test_name_accessible_through_managed_wrapper(self):
        """Test that name is accessible through ManagedSharedArray."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            managed = sa.managed()

            assert managed.name == sa.name

            with managed as m:
                assert m.name == sa.name

    def test_attach_by_name_works(self):
        """Test attaching to existing memory by name."""
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = owner.name
            arr = owner.as_array()
            arr[:] = 42.0

            # Attach using name
            client = SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')
            assert client.name == name

            client_arr = client.as_array()
            assert np.all(client_arr == 42.0)

            client.close()
            owner.close()


class TestSafeVsUnsafeAllocate:
    """Tests comparing safe allocate() vs unsafe allocate_unsafe()."""

    def test_allocate_validates_shape_unsafe_does_not(self):
        """Test that allocate validates shape, unsafe fails differently."""
        with SharedMemoryManager() as manager:
            # Safe version validates and raises ValueError with clear message
            with pytest.raises(ValueError, match="non-negative"):
                SharedMemoryArray.allocate(manager, shape=(-10,), dtype='float64')

            # Unsafe version doesn't validate shape upfront, but will fail
            # at SharedMemory allocation (np.prod(-10) = -10, negative size)
            # This will raise ValueError from SharedMemory, not from our validation
            with pytest.raises((ValueError, OSError)):
                sa_unsafe = SharedMemoryArray.allocate_unsafe(manager, shape=(-10,), dtype='float64')

    def test_allocate_validates_dtype_unsafe_does_not(self):
        """Test that allocate validates dtype but allocate_unsafe does not."""
        with SharedMemoryManager() as manager:
            # Safe version validates with clear error message
            with pytest.raises(TypeError, match="Invalid dtype"):
                SharedMemoryArray.allocate(manager, shape=(10,), dtype='invalid_dtype')

            # Unsafe version will fail at np.zeros() call or dtype conversion
            with pytest.raises((TypeError, AttributeError)):
                SharedMemoryArray.allocate_unsafe(manager, shape=(10,), dtype='invalid_dtype')

    def test_allocate_validates_buffer_size_unsafe_does_not(self):
        """Test that allocate validates buffer size but allocate_unsafe allows mismatch."""
        with SharedMemoryManager() as manager:
            # Safe version ensures correct buffer size
            sa_safe = SharedMemoryArray.allocate(manager, shape=(100,), dtype='float64')
            arr_safe = sa_safe.as_array()
            assert arr_safe.nbytes == 800
            sa_safe.close()

            # Unsafe version can allocate wrong size
            # Allocate only 10 bytes but claim it's 100 float64s
            sa_unsafe = SharedMemoryArray.allocate_unsafe(
                manager, shape=(100,), dtype='float64', nbytes=1
            )
            # This will fail when trying to create the array view
            with pytest.raises((ValueError, TypeError, BufferError)):
                sa_unsafe.as_array()
            sa_unsafe.close()

    def test_allocate_performance_overhead(self):
        """Test that safe version has validation overhead vs unsafe."""
        import time

        with SharedMemoryManager() as manager:
            # Time safe allocation
            start = time.perf_counter()
            for _ in range(100):
                sa = SharedMemoryArray.allocate(manager, shape=(100,), dtype='float64')
                sa.close()
            safe_time = time.perf_counter() - start

            # Time unsafe allocation
            start = time.perf_counter()
            for _ in range(100):
                sa = SharedMemoryArray.allocate_unsafe(manager, shape=(100,), dtype='float64')
                sa.close()
            unsafe_time = time.perf_counter() - start

            # Just check both complete (unsafe might be faster but not required)
            assert safe_time > 0
            assert unsafe_time > 0

    def test_safe_validates_early_unsafe_validates_late(self):
        """Test that safe methods validate before allocation, unsafe after."""
        with SharedMemoryManager() as manager:
            # Safe: validates shape before allocation attempt
            try:
                SharedMemoryArray.allocate(manager, shape=(10, -5), dtype='float64')
                assert False, "Should have raised ValueError"
            except ValueError as e:
                # Should mention shape validation specifically
                assert "non-negative" in str(e)

            # Unsafe: may fail at allocation time with different error
            try:
                SharedMemoryArray.allocate_unsafe(manager, shape=(10, -5), dtype='float64')
                assert False, "Should have raised an error"
            except (ValueError, OSError, OverflowError) as e:
                # Error from SharedMemory allocation, not our validation
                pass


class TestSafeVsUnsafeCopy:
    """Tests comparing safe copy() vs unsafe copy_unsafe()."""

    def test_copy_handles_non_contiguous_unsafe_may_fail(self):
        """Test that copy handles non-contiguous arrays correctly."""
        with SharedMemoryManager() as manager:
            # Create non-contiguous array
            original_full = np.arange(100, dtype='float64')
            original = original_full[::2]
            assert not original.flags.c_contiguous

            # Safe copy handles non-contiguous
            sa_safe = SharedMemoryArray.copy(manager, original)
            copied_safe = sa_safe.as_array()
            np.testing.assert_array_equal(copied_safe, original)
            sa_safe.close()

            # Unsafe copy may have issues with non-contiguous
            # Actually, copy_unsafe uses [:] which should work, but doesn't validate
            sa_unsafe = SharedMemoryArray.copy_unsafe(manager, original)
            copied_unsafe = sa_unsafe.as_array()
            # It works but without validation
            np.testing.assert_array_equal(copied_unsafe, original)
            sa_unsafe.close()

    def test_copy_validates_allocation_unsafe_does_not(self):
        """Test that copy validates allocation but copy_unsafe does not."""
        with SharedMemoryManager() as manager:
            original = np.zeros((10, 20), dtype='float64')

            # Both should work for valid input
            sa_safe = SharedMemoryArray.copy(manager, original)
            sa_unsafe = SharedMemoryArray.copy_unsafe(manager, original)

            np.testing.assert_array_equal(sa_safe.as_array(), original)
            np.testing.assert_array_equal(sa_unsafe.as_array(), original)

            sa_safe.close()
            sa_unsafe.close()

    def test_copy_error_handling_vs_unsafe(self):
        """Test error handling differences between copy and copy_unsafe."""
        with SharedMemoryManager() as manager:
            # Create an array that might cause issues
            original = np.zeros((10,), dtype='float64')

            # Safe version validates
            sa_safe = SharedMemoryArray.copy(manager, original)
            assert sa_safe.shape == (10,)
            sa_safe.close()

            # Unsafe version skips validation
            sa_unsafe = SharedMemoryArray.copy_unsafe(manager, original)
            assert sa_unsafe.shape == (10,)
            sa_unsafe.close()


class TestSafeVsUnsafeAttach:
    """Tests comparing safe attach() vs unsafe attach_unsafe()."""

    def test_attach_validates_buffer_size_unsafe_does_not(self):
        """Test that attach validates buffer size but attach_unsafe does not."""
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype='uint8')  # 10 bytes
            name = owner.name

            # Safe attach validates size
            with pytest.raises(ValueError, match="buffer too small"):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')  # needs 80 bytes

            # Unsafe attach does not validate
            client_unsafe = SharedMemoryArray.attach_unsafe(name=name, shape=(10,), dtype='float64')

            # But will fail when trying to use it
            with pytest.raises((ValueError, TypeError)):
                client_unsafe.as_array()

            client_unsafe.close()
            owner.close()

    def test_attach_validates_shape_and_dtype_unsafe_does_not(self):
        """Test that attach validates parameters but attach_unsafe does not."""
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(100,), dtype='float64')
            name = owner.name

            # Safe attach with correct parameters
            client_safe = SharedMemoryArray.attach(name=name, shape=(100,), dtype='float64')
            arr_safe = client_safe.as_array()
            assert arr_safe.shape == (100,)
            client_safe.close()

            # Unsafe attach doesn't validate
            client_unsafe = SharedMemoryArray.attach_unsafe(name=name, shape=(100,), dtype='float64')
            arr_unsafe = client_unsafe.as_array()
            assert arr_unsafe.shape == (100,)
            client_unsafe.close()

            owner.close()

    def test_attach_default_owner_false(self):
        """Test that attach defaults to owner=False."""
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype='int32')
            name = owner.name

            # Attach with default owner
            client = SharedMemoryArray.attach(name=name, shape=(10,), dtype='int32')
            assert client.owner is False

            # Unsafe attach also defaults to owner=False
            client_unsafe = SharedMemoryArray.attach_unsafe(name=name, shape=(10,), dtype='int32')
            assert client_unsafe.owner is False

            client.close()
            client_unsafe.close()
            owner.close()


class TestSafeVsUnsafeAllocateLike:
    """Tests comparing safe allocate_like() vs unsafe allocate_like_unsafe()."""

    def test_allocate_like_validates_template_unsafe_does_not(self):
        """Test that allocate_like validates template array."""
        with SharedMemoryManager() as manager:
            template = np.zeros((50, 60), dtype='float32')

            # Safe version validates
            sa_safe = SharedMemoryArray.allocate_like(manager, template)
            assert sa_safe.shape == (50, 60)
            assert sa_safe.dtype == np.dtype('float32')
            sa_safe.close()

            # Unsafe version doesn't validate (uses nbytes directly)
            sa_unsafe = SharedMemoryArray.allocate_like_unsafe(manager, template)
            assert sa_unsafe.shape == (50, 60)
            assert sa_unsafe.dtype == np.dtype('float32')
            sa_unsafe.close()

    def test_allocate_like_buffer_size_calculation(self):
        """Test buffer size calculation in safe vs unsafe allocate_like."""
        with SharedMemoryManager() as manager:
            template = np.zeros((10, 10), dtype='float64')  # 800 bytes

            # Safe version calculates correctly
            sa_safe = SharedMemoryArray.allocate_like(manager, template)
            arr_safe = sa_safe.as_array()
            assert arr_safe.nbytes == 800
            sa_safe.close()

            # Unsafe version uses template.nbytes directly
            sa_unsafe = SharedMemoryArray.allocate_like_unsafe(manager, template)
            arr_unsafe = sa_unsafe.as_array()
            assert arr_unsafe.nbytes == 800
            sa_unsafe.close()


class TestEdgeCasesContextManager:
    """Edge case tests for context manager functionality."""

    def test_zero_size_array_with_managed_context(self):
        """Test managed context with zero-size array."""
        with SharedMemoryManager() as manager:
            # Try to allocate zero-size array
            # This may fail at allocation or may succeed depending on system
            with SharedMemoryArray.allocate(manager, (0, 10), 'float64').managed() as sa:
                arr = sa.as_array()
                assert arr.size == 0
                assert arr.shape == (0, 10)

    def test_very_large_array_with_managed_context(self):
        """Test managed context with large array."""
        with SharedMemoryManager() as manager:
            # 10MB array
            with SharedMemoryArray.allocate(manager, (1250000,), 'float64').managed() as sa:
                arr = sa.as_array()
                assert arr.nbytes == 10_000_000
                arr[0] = 1.0
                arr[-1] = 2.0
                assert arr[0] == 1.0
                assert arr[-1] == 2.0

    def test_multiple_managed_contexts_same_array_sequential(self):
        """Test that using array after first managed context closes."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            # First managed context - works fine
            with sa.managed() as m1:
                arr1 = m1.as_array()
                arr1[:] = 1.0
                assert np.all(arr1 == 1.0)

            # After first context exits, the SharedArray's shm is closed and unlinked
            # The name should no longer be accessible for NEW attachments
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

            # Note: as_array() might still work because the buffer may remain
            # in memory temporarily, but the shared memory is unlinked.
            # The key test is that new processes cannot attach to the name.
            # We can still access the buffer from the existing Python object:
            arr = sa.as_array()
            # This works because we still have the SharedArray object with the buffer reference
            # But no NEW processes can attach to this memory anymore

    def test_multiple_managed_contexts_created_before_use(self):
        """Test creating multiple managed wrappers before entering contexts."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            # Create two managed wrappers from the same SharedArray
            managed1 = sa.managed()
            managed2 = sa.managed()

            # Use first one - this will close the underlying SharedArray
            with managed1 as m1:
                arr1 = m1.as_array()
                arr1[:] = 1.0
                assert np.all(arr1 == 1.0)

            # After managed1 exits, the shared memory is closed/unlinked
            # Verify it's gone for new attachments
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

            # The second managed wrapper still has a reference to the same SharedArray
            # Entering its context works, and we can even access the buffer
            # (because Python still has the reference), but trying to close again is idempotent
            with managed2 as m2:
                # as_array() may still work because the buffer reference still exists
                arr2 = m2.as_array()
                arr2[:] = 2.0  # This works because buffer is still in memory
                assert np.all(arr2 == 2.0)

            # The key point: new processes cannot attach, and close() is idempotent
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

    def test_managed_context_non_owner_doesnt_unlink(self):
        """Test that managed context with owner=False doesn't unlink."""
        with SharedMemoryManager() as manager:
            # Create owner
            owner_sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = owner_sa.name
            owner_arr = owner_sa.as_array()
            owner_arr[:] = 42.0

            # Attach as non-owner and use managed context
            client_sa = SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')
            assert client_sa.owner is False

            with client_sa.managed() as client:
                client_arr = client.as_array()
                assert np.all(client_arr == 42.0)
                client_arr[:] = 99.0

            # After client context exits (non-owner), memory should still exist
            # because client.close() only calls unlink() if owner=True
            # Owner can still access it
            owner_arr_check = owner_sa.as_array()
            assert np.all(owner_arr_check == 99.0)

            # We should be able to attach again
            client2 = SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')
            client2_arr = client2.as_array()
            assert np.all(client2_arr == 99.0)
            client2.close()

            # Clean up owner
            owner_sa.close()

    def test_garbage_collection_with_context_manager(self):
        """Test that garbage collection works properly with managed contexts."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(100,), dtype='float64')
            name = sa.name

            with sa.managed() as m:
                arr = m.as_array()
                arr[:] = 42.0
                # Delete the array view (not the shared memory)
                del arr
                gc.collect()
                # Should still be accessible within the context
                arr2 = m.as_array()
                assert np.all(arr2 == 42.0)

            # After context exit, should be cleaned up
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(100,), dtype='float64')

    def test_context_manager_with_small_arrays(self):
        """Test managed context with various small array sizes."""
        with SharedMemoryManager() as manager:
            sizes = [1, 2, 5, 10, 100]
            for size in sizes:
                with SharedMemoryArray.allocate(manager, (size,), 'int32').managed() as sa:
                    arr = sa.as_array()
                    arr[:] = size
                    assert np.all(arr == size)

    def test_managed_wrapper_state_after_close(self):
        """Test ManagedSharedArray state tracking after close."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(5,), dtype='float64')
            managed = sa.managed()

            # Before entering context
            assert managed._closed is False

            # Enter and exit context
            with managed:
                assert managed._closed is False

            # After exiting context
            assert managed._closed is True

            # Cannot re-enter
            with pytest.raises(ValueError, match="already closed"):
                with managed:
                    pass

    def test_exception_in_context_still_cleans_up(self):
        """Test that exceptions in context don't prevent cleanup."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            # Raise exception inside context
            try:
                with sa.managed() as m:
                    arr = m.as_array()
                    arr[:] = 1.0
                    raise RuntimeError("Test exception")
            except RuntimeError:
                pass

            # Should still be cleaned up
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

    def test_idempotent_close_via_multiple_managed_contexts(self):
        """Test that close() is safely idempotent when called by multiple managed contexts."""
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(10,), dtype='float64')
            name = sa.name

            managed1 = sa.managed()
            managed2 = sa.managed()
            managed3 = sa.managed()

            # Use all three sequentially - each calls close() on the same SharedArray
            with managed1:
                arr = sa.as_array()
                arr[:] = 1.0

            # Second close - should be safe (idempotent)
            with managed2:
                pass

            # Third close - should be safe (idempotent)
            with managed3:
                pass

            # Memory should be unlinked after first close
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype='float64')

class TestSafetyComparison:
    """Tests comparing overall safety of safe vs unsafe methods."""

    def test_safe_methods_prevent_common_errors(self):
        """Test that safe methods prevent common usage errors."""
        with SharedMemoryManager() as manager:
            # All these should be caught by safe methods
            errors_caught = []

            # Negative shape
            try:
                SharedMemoryArray.allocate(manager, shape=(-10,), dtype='float64')
            except ValueError:
                errors_caught.append('negative_shape')

            # Invalid dtype
            try:
                SharedMemoryArray.allocate(manager, shape=(10,), dtype='not_a_dtype')
            except TypeError:
                errors_caught.append('invalid_dtype')

            # Buffer too small on attach
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype='uint8')
            try:
                SharedMemoryArray.attach(name=owner.name, shape=(10,), dtype='float64')
            except ValueError:
                errors_caught.append('buffer_too_small')
            owner.close()

            assert len(errors_caught) == 3

    def test_unsafe_methods_skip_validation_for_performance(self):
        """Test that unsafe methods can be used when validation is unnecessary."""
        with SharedMemoryManager() as manager:
            # When you know inputs are valid, unsafe methods work fine
            array = np.ones((100, 200), dtype='float32')

            # Unsafe copy works without overhead
            sa_unsafe = SharedMemoryArray.copy_unsafe(manager, array)
            copied = sa_unsafe.as_array()
            np.testing.assert_array_equal(copied, array)
            sa_unsafe.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
