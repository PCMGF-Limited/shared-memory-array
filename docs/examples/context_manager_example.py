from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool
from tqdm import tqdm

from shmanager.shmanager_numpy import SharedArray


def sum_row(xsi):
    xs, i = xsi
    x = xs.as_array()

    time.sleep(0.01)
    return x[i].sum()


def doshared(show_progress_bar=True):
    n = 1000
    with SharedMemoryManager() as manager:
        xs = SharedArray.allocate(manager, shape=[n, n], dtype=int)
        x = xs.as_array()
        x.fill(1)
        with Pool() as pool:
            print('Initing')
            steps = ((xs, i) for i in range(n))
            print('Startin...')
            result_generator = pool.imap(sum_row, steps)
            print('Computin...')
            if show_progress_bar:
                result_generator = tqdm(result_generator, total=n, position=0, leave=True)
            r = sum(result_generator)
            print('Done.')
            return r


if __name__ == '__main__':
    print("R:", doshared())