
import imageio  # pip insdtall imageio imageio-ffmpeg
import os


def decode_gif_using_imageio(data):
    temp_filename = "/tmp/iotools_temp.gif"
    with open(temp_filename, "wb") as f:
        f.write(data)
    reader = imageio.get_reader(temp_filename)
    data_decoded = [im for im in reader]
    os.remove(temp_filename)
    return data_decoded


def encode_gif_using_imageio(data, fps=30):
    temp_filename = "/tmp/iotools_temp.gif"
    duration = 1000 * 1 / fps
    writer = imageio.get_writer(temp_filename, duration=duration)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_gif_using_imageio(openfile):
    reader = imageio.get_reader(openfile, format="gif")
    return [im for im in reader]


def write_gif_using_imageio(openfile, data, fps=30):
    temp_filename = "/tmp/iotools_temp.gif"
    duration = 1000 * 1 / fps
    writer = imageio.get_writer(temp_filename, duration=duration)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    openfile.write(data_encoded)


def load_gif_using_imageio(filename):
    reader = imageio.get_reader(filename)
    return [im for im in reader]


def save_gif_using_imageio(filename, data, fps=30):
    duration = 1000 * 1 / fps
    writer = imageio.get_writer(filename, duration=duration)
    for frame in data:
        writer.append_data(frame)
    writer.close()
