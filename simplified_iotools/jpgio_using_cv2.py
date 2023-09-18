
import cv2  # pip install opencv-python
import os
import random
import string


def decode_jpg_using_cv2(data):
    temp_filename = "/tmp/iotools_" + "".join(random.choice(string.ascii_letters) for i in range(8))
    with open(temp_filename, "wb") as f:
        f.write(data)
    data_decoded = cv2.imread(temp_filename)
    os.remove(temp_filename)
    return data_decoded


def encode_jpg_using_cv2(data):
    temp_filename = "/tmp/iotools_{}.jpg".format("".join(random.choice(string.ascii_letters) for i in range(8)))
    cv2.imwrite(temp_filename, data)
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_jpg_using_cv2(openfile):
    data = openfile.read()
    temp_filename = "/tmp/iotools_" + "".join(random.choice(string.ascii_letters) for i in range(8))
    with open(temp_filename, "wb") as f:
        f.write(data)
    data_decoded = cv2.imread(temp_filename)
    os.remove(temp_filename)
    return data_decoded


def write_jpg_using_cv2(openfile, data):
    temp_filename = "/tmp/iotools_{}.jpg".format("".join(random.choice(string.ascii_letters) for i in range(8)))
    cv2.imwrite(temp_filename, data)
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    openfile.write(data_encoded)


def load_jpg_using_cv2(filename):
    return cv2.imread(filename)


def save_jpg_using_cv2(filename, data):
    cv2.imwrite(filename, data)
