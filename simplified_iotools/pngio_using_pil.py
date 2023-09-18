
from PIL import Image  # pip install Pillow
from io import BytesIO


def decode_png_using_pil(data):
    filelike = BytesIO(data)
    image = Image.open(filelike)
    image.load()
    filelike.close()
    return image


def encode_png_using_pil(data):
    filelike = BytesIO()
    data.save(filelike, format="png")
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_png_using_pil(openfile):
    image = Image.open(openfile)
    image.load()
    return image


def write_png_using_pil(openfile, data):
    data.save(openfile, format="png")


def load_png_using_pil(filename):
    image = Image.open(filename)
    image.load()
    return image


def save_png_using_pil(filename, data):
    data.save(filename)
