import numpy as np
from rng import RNG

def choice(
    arr: np.ndarray,
    rng: RNG,
    size: int = 1,
) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    rng : RNG
        Random Number Generator object
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # I implement the fisher yates algorithm for randomly drawing unique elements from list
    # source: Wikipedia (https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
    N = len(arr)
    if size > N:
        raise Exception("requested size is larger than array length")

    chosen = []  # instantiate list in which to store chosen samples
    for i in range(size):
        chosen_indx = rng.int((0, N - 1))  # draw a random index
        chosen.append(arr[chosen_indx])
        arr[[i, N - 1]] = arr[[N - 1, i]]  # put drawn sample at N-1th index
        N -= 1  # shorten the amount of indices which can be drawn to avoid drawing same element twice

    return np.array(chosen)
