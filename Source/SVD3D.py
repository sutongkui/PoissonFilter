import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

import common

CacheDict = {}

def ConstructKernel3D(I, J, K, Iter, KernelParas):
    CachedValue = CacheDict.get((I, J, K, Iter))
    if CachedValue is not None:
        return CachedValue

    Kernel = np.zeros((KernelParas.KernelSize, KernelParas.KernelSize, KernelParas.KernelSize), dtype="float64")
    if Iter == 0:
        return Kernel
    else:
        X_IMinus1_J_KMinus1 = ConstructKernel3D(I - 1, J, K , Iter - 1, KernelParas)
        X_IPlus1_J_KMinus1 = ConstructKernel3D(I + 1, J, K, Iter - 1, KernelParas)
        X_I_JMinus1_KMinus1 = ConstructKernel3D(I, J - 1, K, Iter - 1, KernelParas)
        X_I_JPlus1_KMinus1 = ConstructKernel3D(I, J + 1, K, Iter - 1, KernelParas)
        X_IMinus1_J_KPlus1 = ConstructKernel3D(I, J, K - 1, Iter - 1, KernelParas)
        X_IPlus1_J_KPlus1 = ConstructKernel3D(I, J, K + 1, Iter - 1, KernelParas)
        Kernel[I][J][K] = 1

        Ret = (X_IMinus1_J_KMinus1 + X_IPlus1_J_KMinus1 + X_I_JMinus1_KMinus1 + X_I_JPlus1_KMinus1 
                + X_IMinus1_J_KPlus1 + X_IPlus1_J_KPlus1 + KernelParas.alpha * Kernel) / KernelParas.belta
        CacheDict[(I, J, K, Iter)] = Ret
        return Ret
		
def JacobiKernel3D(Ite, alpha, belta):
    KernelSize = 2 * Ite - 1
    CenterIndex = KernelSize // 2
    KernelParas = common.Paras(KernelSize, alpha, belta)
    Kernel = ConstructKernel3D(CenterIndex, CenterIndex, CenterIndex, Ite, KernelParas)
    return Kernel
	
def SVD3D(Kernel, RankSize):
    X = tl.tensor(Kernel)
    CPTensor = parafac(X, rank=RankSize,normalize_factors=True)
    return CPTensor


if __name__=="__main__":

    IterationCount = 3

    # Pressure
    #alpha, belta = common.ComputeInverseParas(3)

    # Diffusion
    alpha, belta = common.ComputeForwardParas(3)

    Kernel = JacobiKernel3D(IterationCount, alpha, belta)
    print("Kernel", Kernel)

    rank = 1
    CPTensor = SVD3D(Kernel, rank)

    print("CPTensor.Weights", CPTensor.weights)

    # Reconstruction
    recon = np.zeros(Kernel.shape)
    for i in range(rank):
        recon += np.einsum('ir,jr,kr->ijk', CPTensor.factors[0][:,i:i+1], CPTensor.factors[1][:,i:i+1], CPTensor.factors[2][:,i:i+1]) * CPTensor.weights[i]
    # print("recon", recon)

    # Compute error
    error = common.ComputeError(Kernel, recon)
    print("error",error)

    # print factors
    print("CPTensor.factors[0] first column", CPTensor.factors[0][:,0:1])
    # print("CPTensor.factors[1] first column", CPTensor.factors[1][:,0:1])
    # print("CPTensor.factors[2] first column", CPTensor.factors[2][:,0:1])