
import imageio  # pip install imageio
from io import BytesIO


def decode_png_using_imageio(data):
    filelike = BytesIO(data)
    ret = imageio.v3.imread(filelike)
    filelike.close()
    return ret


def encode_png_using_imageio(data):
    filelike = BytesIO()
    imageio.imsave(filelike, data, format="png")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_png_using_imageio(openfile):
    return imageio.v3.imread(openfile)


def write_png_using_imageio(openfile, data):
    imageio.imsave(openfile, data, format="png")


def load_png_using_imageio(filename):
    return imageio.v3.imread(filename)


def save_png_using_imageio(filename, data):
    imageio.imsave(filename, data)
