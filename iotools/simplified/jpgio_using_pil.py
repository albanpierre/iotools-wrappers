
from PIL import Image  # pip install Pillow
from io import BytesIO


def decode_jpg_using_pil(data):
    filelike = BytesIO(data)
    image = Image.open(filelike)
    image.load()
    filelike.close()
    return image


def encode_jpg_using_pil(data):
    filelike = BytesIO()
    data.save(filelike, format="jpeg")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_jpg_using_pil(openfile):
    image = Image.open(openfile)
    image.load()
    return image


def write_jpg_using_pil(openfile, data):
    data.save(openfile, format="jpeg")


def load_jpg_using_pil(filename):
    image = Image.open(filename)
    image.load()
    return image


def save_jpg_using_pil(filename, data):
    data.save(filename)
