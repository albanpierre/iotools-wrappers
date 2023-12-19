
import imageio  # pip install imageio
from io import BytesIO


def decode_png(data):
    filelike = BytesIO(data)
    ret = imageio.v3.imread(filelike)
    filelike.close()
    return ret


def encode_png(data):
    filelike = BytesIO()
    imageio.imsave(filelike, data, format="png")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_png(openfile):
    return imageio.v3.imread(openfile)


def write_png(openfile, data):
    imageio.imsave(openfile, data, format="png")


def load_png(filename):
    return imageio.v3.imread(filename)


def save_png(filename, data):
    imageio.imsave(filename, data)
