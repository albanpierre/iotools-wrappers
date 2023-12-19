
import skimage.io as skimageio  # pip install scikit-image
from io import BytesIO


def decode_image_using_skimage(data):
    filelike = BytesIO(data)
    ret = skimageio.imread(filelike)
    filelike.close()
    return ret


def encode_image_using_skimage(data, format=None):
    filelike = BytesIO()
    skimageio.imsave(filelike, data, extension="." + format)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_image_using_skimage(openfile):
    return skimageio.imread(openfile)


def write_image_using_skimage(openfile, data, format=None):
    skimageio.imsave(openfile, data, extension="." + format)


def load_image_using_skimage(filename):
    return skimageio.imread(filename)


def save_image_using_skimage(filename, data):
    skimageio.imsave(filename, data)
