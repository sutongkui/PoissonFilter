import numpy as np
import os

# Jacobi iteration steps
# Adjust this variable to get what you want
IterationCount = 3

CacheDict = {}

def DivergenceIter(I, J, K):
    CachedValue = CacheDict.get((I, J, K))
    if CachedValue is not None:
        return CachedValue
    KernelSize = 2 * IterationCount - 1
    Kernel = np.zeros((KernelSize, KernelSize))
    if K == 0:
        return Kernel
    else:
        XIMinus1J = DivergenceIter(I - 1, J, K - 1)
        XIPlus1J = DivergenceIter(I + 1, J, K - 1)
        XIJMinus1 = DivergenceIter(I, J - 1, K - 1)
        XIJPlus1 = DivergenceIter(I, J + 1, K - 1)
        Kernel[I][J] = 1
        Ret = XIMinus1J * 0.25 + XIPlus1J * 0.25 + XIJMinus1 * 0.25 + XIJPlus1 * 0.25 - Kernel * 0.25
        CacheDict[(I, J, K)] = Ret
        return Ret
		
def DivergenceKenel(IterationCount):
    KernelSize = 2 * IterationCount - 1
    CenterIndex = KernelSize // 2
    Kernel = DivergenceIter(CenterIndex, CenterIndex, IterationCount)
    return Kernel
	
def Svd(Kernel):
    U, Sigma, VT = np.linalg.svd(Kernel)
    return  U, Sigma, VT

def ComputeError(matrix1, matrix2):
    difference = matrix1 - matrix2
    error = np.sqrt(np.sum(np.square(difference)))
    return error


if __name__=="__main__":

    Kernel = DivergenceKenel(IterationCount)
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