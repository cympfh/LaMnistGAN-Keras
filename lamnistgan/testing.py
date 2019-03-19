import numpy

from PIL import Image


def imgsave(X, path, n=10):
    num = min(len(X), n)
    arrs = []
    for i in range(num):
        arr = numpy.clip(X[i], 0, 1)
        arr = (arr * 255).reshape((28, 28)).astype(numpy.uint8)
        arrs.append(arr)
    img = Image.fromarray(numpy.hstack(arrs))
    img.save(path)
