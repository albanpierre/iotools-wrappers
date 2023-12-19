
import imageio  # pip install imageio
from io import BytesIO


def decode_image(data):
    filelike = BytesIO(data)
    ret = imageio.v3.imread(filelike)
    filelike.close()
    return ret


def encode_image(data, format=None):
    filelike = BytesIO()
    imageio.imsave(filelike, data, format=format)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_image(openfile):
    return imageio.v3.imread(openfile)


def write_image(openfile, data, format=None):
    imageio.imsave(openfile, data, format=format)


def load_image(filename):
    return imageio.v3.imread(filename)


def save_image(filename, data):
    imageio.imsave(filename, data)
