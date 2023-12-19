
from PIL import Image  # pip install Pillow
from io import BytesIO


def decode_image_using_pil(data):
    filelike = BytesIO(data)
    image = Image.open(filelike)
    image.load()
    filelike.close()
    return image


def encode_image_using_pil(data, format=None):
    filelike = BytesIO()
    data.save(filelike, format=format.replace("jpg", "jpeg"))
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_image_using_pil(openfile):
    image = Image.open(openfile)
    image.load()
    return image


def write_image_using_pil(openfile, data, format=None):
    data.save(openfile, format=format.replace("jpg", "jpeg"))


def load_image_using_pil(filename):
    image = Image.open(filename)
    image.load()
    return image


def save_image_using_pil(filename, data):
    data.save(filename)
