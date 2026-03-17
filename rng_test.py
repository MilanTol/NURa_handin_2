from rng import RNG

rng = RNG(seed=1)

# in this code snippet I will test the random number generator.
# for this I follow this article: https://random.tastemaker.design/goodness-of-fit


def Chi2(observed: list, expected: list) -> float:
    """
    returns Chi-squared test for observed vs expected data
    """
    chi2 = 0
    for i in range(len(observed)):
        chi2 += (observed[i] - expected[i]) ** 2 / expected[i]
    return chi2
