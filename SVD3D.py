import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

IterationCount = 3
CacheDict = {}

def DivergenceIter3D(I, J, K, Iter):
    CachedValue = CacheDict.get((I, J, K, Iter))
    if CachedValue is not None:
        return CachedValue
    KernelSize = 2 * IterationCount - 1
    Kernel = np.zeros((KernelSize, KernelSize, KernelSize), dtype="float64")
    if Iter == 0:
        return Kernel
    else:
        XIMinus1JKMinus1 = DivergenceIter3D(I - 1, J, K , Iter - 1)
        XIPlus1JKMinus1 = DivergenceIter3D(I + 1, J, K, Iter - 1)
        XIJMinus1KMinus1 = DivergenceIter3D(I, J - 1, K, Iter - 1)
        XIJPlus1KMinus1 = DivergenceIter3D(I, J + 1, K, Iter - 1)

        XIMinus1JKPlus1 = DivergenceIter3D(I, J, K - 1, Iter - 1)
        XIPlus1JKPlus1 = DivergenceIter3D(I, J, K + 1, Iter - 1)
        Kernel[I][J][K] = 1

        OneDivSix = 1.0 / 6.0;
        Ret = XIMinus1JKMinus1 * OneDivSix + XIPlus1JKMinus1 * OneDivSix + XIJMinus1KMinus1 * OneDivSix + XIJPlus1KMinus1 * OneDivSix  + XIMinus1JKPlus1 * OneDivSix + XIPlus1JKPlus1 * OneDivSix - Kernel * OneDivSix
        CacheDict[(I, J, K, Iter)] = Ret
        return Ret
		
def DivergenceKenel(IterationCount):
    KernelSize = 2 * IterationCount - 1
    CenterIndex = KernelSize // 2
    Kernel = DivergenceIter3D(CenterIndex, CenterIndex, CenterIndex, IterationCount)
    return Kernel
	
def Svd(Kernel, RankSize):
    X = tl.tensor(Kernel)
    CPTensor = parafac(X, rank=RankSize,normalize_factors=True)
    return CPTensor

def ComputeError(Tensor1, Tensor2):
    difference = Tensor1 - Tensor2
    error = np.sqrt(np.sum(np.square(difference)))
    return error

if __name__=="__main__":

    Kernel = DivergenceKenel(IterationCount)
    print("Kernel", Kernel)

    rank = 1
    CPTensor = Svd(Kernel, rank)

    print("CPTensor.Weights", CPTensor.weights)

    # Reconstruction
    recon = np.zeros(Kernel.shape)
    for i in range(rank):
        recon += np.einsum('ir,jr,kr->ijk', CPTensor.factors[0][:,i:i+1], CPTensor.factors[1][:,i:i+1], CPTensor.factors[2][:,i:i+1]) * CPTensor.weights[i]
    # print("recon", recon)

    # Compute error
    error = ComputeError(Kernel, recon)
    print("error",error)

    # print factors
    print("CPTensor.factors[0] first column", CPTensor.factors[0][:,0:1])
    # print("CPTensor.factors[1] first column", CPTensor.factors[1][:,0:1])
    # print("CPTensor.factors[2] first column", CPTensor.factors[2][:,0:1])