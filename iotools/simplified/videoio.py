
import imageio  # pip insdtall imageio imageio-ffmpeg
import os


def decode_video(data, format="mp4"):
    temp_filename = "/tmp/iotools_temp"
    with open(temp_filename, "wb") as f:
        f.write(data)
    reader = imageio.get_reader(temp_filename, format=format)
    data_decoded = [im for im in reader]
    os.remove(temp_filename)
    return data_decoded


def encode_video(data, format="mp4", fps=30):
    temp_filename = "/tmp/iotools_temp." + format
    if format == "gif":
        kwargs = {"duration": 1000 * 1 / fps}
    else:
        kwargs = {"fps": fps}
    writer = imageio.get_writer(temp_filename, format=format, **kwargs)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_video(openfile, format="mp4"):
    reader = imageio.get_reader(openfile, format=format)
    return [im for im in reader]


def write_video(openfile, data, format="mp4", fps=30):
    temp_filename = "/tmp/iotools_temp." + format
    if format == "gif":
        kwargs = {"duration": 1000 * 1 / fps}
    else:
        kwargs = {"fps": fps}
    writer = imageio.get_writer(temp_filename, format=format, **kwargs)
    for frame in data:
        writer.append_data(frame)
    writer.close()
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    openfile.write(data_encoded)


def load_video(filename):
    reader = imageio.get_reader(filename)
    return [im for im in reader]


def save_video(filename, data, format="mp4", fps=30):
    if format == "gif":
        kwargs = {"duration": 1000 * 1 / fps}
    else:
        kwargs = {"fps": fps}
    writer = imageio.get_writer(filename, format=format, **kwargs)
    for frame in data:
        writer.append_data(frame)
    writer.close()
