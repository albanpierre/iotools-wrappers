
import cv2  # pip install opencv-python
import os


def decode_mp4_using_cv2(data):
    temp_filename = "/tmp/iotools_temp.mp4"
    with open(temp_filename, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(temp_filename)
    video = []
    ret = True
    while ret:
        ret, frame = cap.read()
        video.append(frame)
    cap.release()
    data_decoded = [frame[:, :, 2::-1] for frame in video[:-1]]
    os.remove(temp_filename)
    return data_decoded


def encode_mp4_using_cv2(data, codec="mp4v", fps=30):
    temp_filename = "/tmp/iotools_temp.mp4"
    height = len(data[0])
    width = len(data[0][0])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
    for frame in data:
        out.write(frame[:, :, 2::-1])
    out.release()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_mp4_using_cv2(openfile):
    data = openfile.read()
    temp_filename = "/tmp/iotools_temp.mp4"
    with open(temp_filename, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(temp_filename)
    video = []
    ret = True
    while ret:
        ret, frame = cap.read()
        video.append(frame)
    cap.release()
    data_decoded = [frame[:, :, 2::-1] for frame in video[:-1]]
    os.remove(temp_filename)
    return data_decoded


def write_mp4_using_cv2(openfile, data, codec="mp4v", fps=30):
    temp_filename = "/tmp/iotools_temp.mp4"
    height = len(data[0])
    width = len(data[0][0])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
    for frame in data:
        out.write(frame[:, :, 2::-1])
    out.release()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    openfile.write(data_encoded)


def load_mp4_using_cv2(filename):
    cap = cv2.VideoCapture(filename)
    video = []
    ret = True
    while ret:
        ret, frame = cap.read()
        video.append(frame)
    cap.release()
    return [frame[:, :, 2::-1] for frame in video[:-1]]


def save_mp4_using_cv2(filename, data, codec="mp4v", fps=30):
    height = len(data[0])
    width = len(data[0][0])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in data:
        out.write(frame[:, :, 2::-1])
    out.release()
