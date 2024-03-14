# Ref:
# Compact Poisson Filters for Fast Fluid Simulation: https://dl.acm.org/doi/pdf/10.1145/3528233.3530737
# supplementary material: https://github.com/ubisoft/ubisoft-laforge-Poisson-Filters/blob/main/docs/paper/supplementary.pdf

import numpy as np
import os

import common

CacheDict = {}

def ConstructKernel2D(I, J, K, KernelParas):
    CachedValue = CacheDict.get((I, J, K))
    if CachedValue is not None:
        return CachedValue

    Kernel = np.zeros((KernelParas.KernelSize, KernelParas.KernelSize))
    if K == 0:
        return Kernel
    else:
        X_IMinus1_J = ConstructKernel2D(I - 1, J, K - 1, KernelParas)
        X_IPlus1_J = ConstructKernel2D(I + 1, J, K - 1, KernelParas)
        X_I_JMinus1 = ConstructKernel2D(I, J - 1, K - 1, KernelParas)
        X_I_JPlus1 = ConstructKernel2D(I, J + 1, K - 1, KernelParas)
        Kernel[I][J] = 1
        Ret = (X_IMinus1_J + X_IPlus1_J + X_I_JMinus1 + X_I_JPlus1 + KernelParas.alpha * Kernel) / KernelParas.belta
        CacheDict[(I, J, K)] = Ret
        return Ret
		
def JacobiKernel2D(Ite, alpha, belta):
    KernelSize = 2 * Ite - 1
    CenterIndex = KernelSize // 2
    KernelParas = common.Paras(KernelSize, alpha, belta)
    Kernel = ConstructKernel2D(CenterIndex, CenterIndex, Ite, KernelParas)
    return Kernel
	
def SVD2D(Kernel):
    U, Sigma, VT = np.linalg.svd(Kernel)
    return  U, Sigma, VT

 

if __name__=="__main__":

    # Jacobi iteration steps
    # Adjust this variable to get what you want
    IterationCount = 2

    #alpha, belta = common.ComputeInverseParas(2)
    alpha, belta = common.ComputeForwardParas(2)
    Kernel = JacobiKernel2D(IterationCount, alpha, belta)

    print("Original Kernel:",Kernel)

    # Decomposition
    U, Sigma, VT = SVD2D(Kernel)

    # Reconstruction
    Recon = np.zeros((Kernel.shape[0], Kernel.shape[1]))
    singularNum = Sigma.shape[0]

    for i in range(singularNum):
        Recon += U[:,i:i+1] @ VT[i:i+1,:] * Sigma[i]
        
    print("Recon:",Recon)

    # Compute error
    error = common.ComputeError(Kernel, Recon)
    print("error: ", error)