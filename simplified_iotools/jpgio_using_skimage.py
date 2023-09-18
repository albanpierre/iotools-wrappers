
import skimage.io as skimageio  # pip install scikit-image
from io import BytesIO


def decode_jpg_using_skimage(data):
    filelike = BytesIO(data)
    ret = skimageio.imread(filelike)
    filelike.close()
    return ret


def encode_jpg_using_skimage(data):
    filelike = BytesIO()
    skimageio.imsave(filelike, data, extension=".jpg")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_jpg_using_skimage(openfile):
    return skimageio.imread(openfile)


def write_jpg_using_skimage(openfile, data):
    skimageio.imsave(openfile, data, extension=".jpg")


def load_jpg_using_skimage(filename):
    return skimageio.imread(filename)


def save_jpg_using_skimage(filename, data):
    skimageio.imsave(filename, data)
