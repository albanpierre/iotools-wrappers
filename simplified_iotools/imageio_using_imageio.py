
import imageio  # pip install imageio
from io import BytesIO


def decode_image_using_imageio(data):
    filelike = BytesIO(data)
    ret = imageio.v3.imread(filelike)
    filelike.close()
    return ret


def encode_image_using_imageio(data, format=None):
    filelike = BytesIO()
    imageio.imsave(filelike, data, format=format)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_image_using_imageio(openfile):
    return imageio.v3.imread(openfile)


def write_image_using_imageio(openfile, data, format=None):
    imageio.imsave(openfile, data, format=format)


def load_image_using_imageio(filename):
    return imageio.v3.imread(filename)


def save_image_using_imageio(filename, data):
    imageio.imsave(filename, data)
