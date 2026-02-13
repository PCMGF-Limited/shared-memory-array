from multiprocessing.managers import SharedMemoryManager

from shmanager import SharedArray

with SharedMemoryManager() as manager:
    with SharedArray.allocate(manager, (100,), 'float64').managed() as sa:

        shm, shape, dtype, owner = sa
        print('shape: ', shape)
        print('dtype: ', dtype)
        print('is owner: ', owner)

        arr = sa.as_array()
        arr[:] = 1.0
        # Automatic cleanup