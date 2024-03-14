# Ref:
# Compact Poisson Filters for Fast Fluid Simulation: https://dl.acm.org/doi/pdf/10.1145/3528233.3530737
# supplementary material: https://github.com/ubisoft/ubisoft-laforge-Poisson-Filters/blob/main/docs/paper/supplementary.pdf

import numpy as np
import os

CacheDict = {}

class Paras:
    def __init__(self, size, alpha, belta):
        self.KernelSize = size
        self.alpha = alpha
        self.belta = belta

def ConstructKernel(I, J, K, KernelParas):
    CachedValue = CacheDict.get((I, J, K))
    if CachedValue is not None:
        return CachedValue

    Kernel = np.zeros((KernelParas.KernelSize, KernelParas.KernelSize))
    if K == 0:
        return Kernel
    else:
        X_IMinus1_J = ConstructKernel(I - 1, J, K - 1, KernelParas)
        X_IPlus1_J = ConstructKernel(I + 1, J, K - 1, KernelParas)
        X_I_JMinus1 = ConstructKernel(I, J - 1, K - 1, KernelParas)
        X_I_JPlus1 = ConstructKernel(I, J + 1, K - 1, KernelParas)
        Kernel[I][J] = 1
        Ret = (X_IMinus1_J + X_IPlus1_J + X_I_JMinus1 + X_I_JPlus1 + KernelParas.alpha * Kernel) / KernelParas.belta
        CacheDict[(I, J, K)] = Ret
        return Ret
		
def JacobiKernel(Ite, alpha, belta):
    KernelSize = 2 * Ite - 1
    CenterIndex = KernelSize // 2
    KernelParas = Paras(KernelSize, alpha, belta)
    Kernel = ConstructKernel(CenterIndex, CenterIndex, Ite, KernelParas)
    return Kernel
	
def Svd(Kernel):
    U, Sigma, VT = np.linalg.svd(Kernel)
    return  U, Sigma, VT

def ComputeError(matrix1, matrix2):
    difference = matrix1 - matrix2
    error = np.sqrt(np.sum(np.square(difference)))
    return error

# Forward problem, diffusion in fluid simulation
def ComputeForwardParas():

    viscosity = 10.0

    # 30 frames per second, delta_t = 1.0/30
    delta_t = 0.033

    # Fix delta_x = 1 m
    delta_x = 1.0
    
    # 2D
    dimension = 2

    alpha = delta_x**2 / (viscosity * delta_t)
    belta = 2 * dimension + alpha

    return alpha, belta


# Inverse problem, pressure solver in in fluid simulation
def ComputeInverseParas():

    # Fix delta_x = 1 m
    delta_x = 1.0

    # 2D
    dimension = 2

    alpha = - delta_x**2
    belta = 2.0 * dimension

    return alpha, belta
 

if __name__=="__main__":

    # Jacobi iteration steps
    # Adjust this variable to get what you want
    IterationCount = 2

    #alpha, belta = ComputeInverseParas()
    alpha, belta = ComputeForwardParas()
    Kernel = JacobiKernel(IterationCount, alpha, belta)

    print("Original Kernel:",Kernel)

    # Decomposition
    U, Sigma, VT = Svd(Kernel)

    # Reconstruction
    Recon = np.zeros((Kernel.shape[0], Kernel.shape[1]))
    singularNum = Sigma.shape[0]

    for i in range(singularNum):
        Recon += U[:,i:i+1] @ VT[i:i+1,:] * Sigma[i]
        
    print("Recon:",Recon)

    # Compute error
    error = ComputeError(Kernel, Recon)
    print("error: ", error)