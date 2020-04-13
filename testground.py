import numpy as np


def fieldTransformer(data: np.ndarray, areaSize):
    mapSize = data.shape[-1]
    tmpSize = list(data.shape)
    # Index : (100,100) => (98,98) [0,97]
    sizeAfterTransform = mapSize - areaSize + 1
    tmpSize[-1] = sizeAfterTransform
    tmpSize[-2] = sizeAfterTransform
    tmpSize[-3] = 1
    tmpMin = np.zeros(shape=tmpSize)
    tmpMax = np.zeros(shape=tmpSize)

    for i in range(sizeAfterTransform):
        for j in range(sizeAfterTransform):
            tmp1 = np.sum(data[..., i:i + 3, j:j + 3], axis=(-1, -2))

            tmpMin[..., i, j] = np.min(tmp1, axis=-1, keepdims=True)
            tmpMax[..., i, j] = np.max(tmp1, axis=-1, keepdims=True)

    return tmpMin, tmpMax


if __name__ == "__main__":
    testData = np.arange(1800).reshape(3, 6, 10, 10)
    resMin, resMax = fieldTransformer(testData, 3)
    print(resMin.shape)
    print(resMin)
    print(resMax.shape)
    print(resMax)
    
