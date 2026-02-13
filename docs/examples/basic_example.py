from multiprocessing.managers import SharedMemoryManager

from shared_memory_array import SharedMemoryArray


with SharedMemoryManager() as manager:
    sa = SharedMemoryArray.allocate(manager, (100,), 'float64')
    shm, shape, dtype, owner = sa

    print('shape: ', shape)
    print('dtype: ', dtype)
    print('is owner: ', owner)

    arr = sa.as_array()
    arr[:] = 1.0
    sa.close()  # Manual cleanup