from multiprocessing.managers import SharedMemoryManager

from shmanager import SharedArray


with SharedMemoryManager() as manager:
    sa = SharedArray.allocate(manager, (100,), 'float64')
    shm, shape, dtype, owner = sa

    print('shape: ', shape)
    print('dtype: ', dtype)
    print('is owner: ', owner)

    arr = sa.as_array()
    arr[:] = 1.0
    sa.close()  # Manual cleanup