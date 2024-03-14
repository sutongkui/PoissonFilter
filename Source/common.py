import numpy as np

class Paras:
    def __init__(self, size, alpha, belta):
        self.KernelSize = size
        self.alpha = alpha
        self.belta = belta


# Forward problem, diffusion in fluid simulation
def ComputeForwardParas(dimension):

    viscosity = 10.0

    # 30 frames per second, delta_t = 1.0/30
    delta_t = 0.033

    # Fix delta_x = 1 m
    delta_x = 1.0

    alpha = delta_x**2 / (viscosity * delta_t)
    belta = 2 * dimension + alpha

    return alpha, belta


# Inverse problem, pressure solver in in fluid simulation
def ComputeInverseParas(dimension):

    # Fix delta_x = 1 m
    delta_x = 1.0

    alpha = - delta_x**2
    belta = 2.0 * dimension

    return alpha, belta


def ComputeError(Tensor1, Tensor2):
    difference = Tensor1 - Tensor2
    error = np.sqrt(np.sum(np.square(difference)))
    return error