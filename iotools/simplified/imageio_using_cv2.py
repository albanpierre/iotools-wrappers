
import cv2  # pip install opencv-python
import numpy as np


def decode_image_using_cv2(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def encode_image_using_cv2(data, format="png"):
    return cv2.imencode("." + format, data)[1].tobytes()


def read_image_using_cv2(openfile):
    data = openfile.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def write_image_using_cv2(openfile, data, format="png"):
    data_encoded = cv2.imencode("." + format, data)[1].tobytes()
    openfile.write(data_encoded)


def load_image_using_cv2(filename):
    return cv2.imread(filename)


def save_image_using_cv2(filename, data):
    cv2.imwrite(filename, data)
