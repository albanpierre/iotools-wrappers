
import skimage.io as skimageio  # pip install scikit-image
from io import BytesIO


def decode_png_using_skimage(data):
    filelike = BytesIO(data)
    ret = skimageio.imread(filelike)
    filelike.close()
    return ret


def encode_png_using_skimage(data):
    filelike = BytesIO()
    skimageio.imsave(filelike, data, extension=".png")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_png_using_skimage(openfile):
    return skimageio.imread(openfile)


def write_png_using_skimage(openfile, data):
    skimageio.imsave(openfile, data, extension=".png")


def load_png_using_skimage(filename):
    return skimageio.imread(filename)


def save_png_using_skimage(filename, data):
    skimageio.imsave(filename, data)
