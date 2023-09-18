
import imageio  # pip insdtall imageio imageio-ffmpeg
import os


def decode_mp4_using_imageio(data):
    temp_filename = "/tmp/iotools_temp.mp4"
    with open(temp_filename, "wb") as f:
        f.write(data)
    reader = imageio.get_reader(temp_filename)
    data_decoded = [im for im in reader]
    os.remove(temp_filename)
    return data_decoded


def encode_mp4_using_imageio(data, fps=30):
    temp_filename = "/tmp/iotools_temp.mp4"
    writer = imageio.get_writer(temp_filename, fps=fps)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_mp4_using_imageio(openfile):
    reader = imageio.get_reader(openfile, format="mp4")
    return [im for im in reader]


def write_mp4_using_imageio(openfile, data, fps=30):
    temp_filename = "/tmp/iotools_temp.mp4"
    writer = imageio.get_writer(temp_filename, fps=fps)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    openfile.write(data_encoded)


def load_mp4_using_imageio(filename):
    reader = imageio.get_reader(filename)
    return [im for im in reader]


def save_mp4_using_imageio(filename, data, fps=30):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in data:
        writer.append_data(frame)
    writer.close()
