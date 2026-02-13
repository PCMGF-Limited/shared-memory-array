"""
“Collisions”: checks that two allocations do not reuse the same shared-memory name, and that attaching a random non-existent name fails.

“Ownership”: asserts attach(...).owner is False, verifies closing a non-owner does not unlink, and verifies closing an owner unlinks and makes future attaches fail.

“Attach/detach + complexity”: tests size/dtype mismatches for both safe (attach) and unsafe (attach_unsafe, allocate_unsafe) paths, including failure modes when buffer sizes don’t match the declared view.
"""
import uuid
import pytest
import numpy as np

from multiprocessing.managers import SharedMemoryManager
from shared_memory_array import SharedMemoryArray


class TestNameHandling:
    def test_two_allocations_have_distinct_names(self):
        with SharedMemoryManager() as manager:
            sa1 = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.uint8)
            sa2 = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.uint8)
            assert sa1.shm.name != sa2.shm.name


    def test_attach_nonexistent_name_raises_filenotfound(self):
        # Use a name that is overwhelmingly unlikely to exist
        name = f"shmanager-test-{uuid.uuid4().hex}"
        with pytest.raises(FileNotFoundError):
            SharedMemoryArray.attach(name=name, shape=(1,), dtype=np.uint8)


class TestOwnershipHandling:
    def test_owner_false_close_does_not_unlink(self):
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.int64)
            name = owner.shm.name

            # Attach from a "different process" perspective
            client = SharedMemoryArray.attach(name=name, shape=(10,), dtype=np.int64)
            assert client.owner is False

            # Closing non-owner must not unlink the shared memory name
            client.close()

            # Should still be attachable (name still exists)
            client2 = SharedMemoryArray.attach(name=name, shape=(10,), dtype=np.int64)
            client2.close()

            # Owner can still read/write
            arr = owner.as_array()
            arr[:] = np.arange(10, dtype=np.int64)
            np.testing.assert_array_equal(arr, np.arange(10, dtype=np.int64))


    def test_owner_close_unlinks_and_future_attach_fails(self):
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.float64)
            name = owner.shm.name

            # close() for owner unlinks in your implementation
            owner.close()

            # After unlink, attaching by name should fail
            with pytest.raises(FileNotFoundError):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype=np.float64)

class TestCloseUnlink:
    def test_unlink_is_idempotent(self):
        with SharedMemoryManager() as manager:
            sa = SharedMemoryArray.allocate(manager, shape=(4,), dtype=np.uint8)
            # Should not raise even if called multiple times
            sa.unlink()
            sa.unlink()
            # close() should also be safe after prior unlink (your close() calls unlink for owner)
            sa.close()

class TestAttach:
    def test_attach_wrong_shape_raises_buffer_too_small(self):
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.float64)
            name = owner.shm.name

            # Requesting a larger view than allocated should be rejected by attach()
            with pytest.raises(ValueError, match="buffer too small"):
                SharedMemoryArray.attach(name=name, shape=(11,), dtype=np.float64)

            owner.close()


    def test_attach_wrong_dtype_raises_buffer_too_small_when_itemsize_differs(self):
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate(manager, shape=(10,), dtype=np.uint8)  # 10 bytes
            name = owner.shm.name

            # float64 would require 80 bytes for shape (10,)
            with pytest.raises(ValueError, match="buffer too small"):
                SharedMemoryArray.attach(name=name, shape=(10,), dtype=np.float64)

            owner.close()


    def test_attach_unsafe_allows_mismatch_but_as_array_raises(self):
        with SharedMemoryManager() as manager:
            owner = SharedMemoryArray.allocate_like_unsafe(manager, np.zeros((10,), dtype=np.uint8))  # 10 bytes
            name = owner.shm.name

            # Unsafe attach does not validate size/dtype/shape
            bad = SharedMemoryArray.attach_unsafe(name=name, shape=(10,), dtype=np.float64)  # would need 80 bytes

            with pytest.raises((ValueError, TypeError)):
                # NumPy should refuse to construct an array view larger than the buffer.
                _ = bad.as_array()

            bad.close()
            owner.close()

class TestAllocateSize:
    def test_allocate_unsafe_with_incorrect_nbytes_breaks_on_as_array(self):
        with SharedMemoryManager() as manager:
            # Allocates only 10 bytes, but declares dtype float64 and shape (10,) => would need 80 bytes.
            sa = SharedMemoryArray.allocate_unsafe(manager, shape=(10,), dtype=np.float64, nbytes=1)

            with pytest.raises((ValueError, TypeError)):
                _ = sa.as_array()

            sa.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
