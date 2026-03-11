import numpy as np
from .rng import RNG

class Distribution:
    def __init__(self, dist:callable, min:float, max:float, args:tuple = (), seed=1):
        """
        Parameters
        dist : callable
            function that is proportional to distribution, must not be negative anywhere.
        min :
            Minimum value for sampling
        max : float
            Maximum value for sampling
        args : tuple, optional
            Arguments of the distribution to sample, passed as args to dist
        seed : int, optional
            Seed used for random number generation

        """
        self.dist = dist
        self.min = min
        self.max = max
        self.args = args
        self.rng = RNG(seed)

    def rejection(self, N_samples:int=1, args:tuple=None) -> np.ndarray:
        """
        samples x values from the distribution using rejection sampling

        Parameters
        Nsamples : int
            Number of samples
        args : tuple, optional
            Arguments of the distribution to sample, passed as args to dist

        Returns
        -------
        sample: ndarray
            Values sampled from dist, shape (Nsamples,)
        """        
        samples = []
        for i in range(N_samples):
            x = self.rng.float((self.min, self.max)) #randomly draw an x in between bounds
            y = self.rng.float() #draw a y in between 0 and 1
            if y > self.dist(x, *args):
                samples.append(x)
        return np.array(samples)




