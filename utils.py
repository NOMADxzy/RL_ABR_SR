import random
sigma = 4
def get_random_gauss_val(bottom, top):
    mean = (bottom + top)/2
    val = random.gauss(mean, sigma)
    while val < bottom or val > top:
        val = random.gauss(mean, sigma)
    return val