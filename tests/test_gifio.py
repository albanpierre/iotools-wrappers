
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.gifio import (
    decode_gif_using_imageio, encode_gif_using_imageio,
    read_gif_using_imageio, write_gif_using_imageio,
    load_gif_using_imageio, save_gif_using_imageio, help_gif_using_imageio,
    decode_gif, encode_gif, read_gif, write_gif, load_gif, save_gif, help_gif
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_GIF = os.path.join(TEST_DATA_PATH, "video", "example_gif_{}.gif")

data0 = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(6)]
for i in range(6):
    for x in range(3):
        for y in range(3):
            data0[i][(i // 3) * 16 + x, (i % 3) * 16 + y, :] = [55, 155, 255]

data = {}
data[0] = data0


def _format_video(vid):
    for i in range(len(vid)):
        if not isinstance(vid[i], np.ndarray):
            vid[i] = np.asarray(vid[i])
        if vid[i].ndim == 2:
            vid[i] = vid[i][:, :, np.newaxis].repeat(4, axis=2)
            vid[i][:, :, 3] = 255
        if vid[i].shape[2] == 3:
            vid[i] = np.concatenate([vid[i], np.full(vid[i].shape[:2] + (1,), 255)], axis=2)
        vid[i] = vid[i].astype(np.uint8)
    return vid


def _video_distance(vid1, vid2):
    vid1 = _format_video(vid1)
    vid2 = _format_video(vid2)
    if len(vid1) != len(vid2):
        return 256
    if vid1[0].shape != vid2[0].shape:
        return 256
    return np.mean([
        np.mean(np.abs(frame1.astype(int) - frame2.astype(int))) for frame1, frame2 in zip(vid1, vid2)
    ])


def _video_distance_per_frame(vid1, vid2):
    vid1 = _format_video(vid1)
    vid2 = _format_video(vid2)
    if len(vid1) != len(vid2):
        return 256
    if vid1[0].shape != vid2[0].shape:
        return 256
    return [np.mean(np.abs(frame1.astype(int) - frame2.astype(int))) for frame1, frame2 in zip(vid1, vid2)]


# +---------------+
# | Using imageio |
# +---------------+


def test_decode_gif_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif_using_imageio(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif_using_imageio(txt, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif_using_imageio(txt, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif_using_imageio(txt, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_encode_gif_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        datai = decode_gif_using_imageio(txt)
        txt = encode_gif_using_imageio(datai)
        ldata = decode_gif_using_imageio(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        datai = decode_gif_using_imageio(txt, which="all")
        txt = encode_gif_using_imageio(datai)
        ldata = decode_gif_using_imageio(txt, which="all")
        assert _video_distance(ldata['video'], data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_gif_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_write_gif_using_imageio():
    filename = FILENAMES_GIF.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        with open(filename2, "rb") as f:
            datai = read_gif_using_imageio(f)
        with open(filename, "wb") as f:
            write_gif_using_imageio(f, datai)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        with open(filename2, "rb") as f:
            datai = read_gif_using_imageio(f, which="all")
        with open(filename, "wb") as f:
            write_gif_using_imageio(f, datai)
        with open(filename, "rb") as f:
            ldata = read_gif_using_imageio(f, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_load_gif_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif_using_imageio(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif_using_imageio(filename, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif_using_imageio(filename, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif_using_imageio(filename, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_save_gif_using_imageio():
    filename = FILENAMES_GIF.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        datai = load_gif_using_imageio(filename2)
        save_gif_using_imageio(filename, datai)
        ldata = load_gif_using_imageio(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        datai = load_gif_using_imageio(filename2, which="all")
        save_gif_using_imageio(filename, datai)
        ldata = load_gif_using_imageio(filename, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_help_gif_using_imageio():
    help_gif_using_imageio()


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_gif():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif(txt, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif(txt, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        ldata = decode_gif(txt, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_encode_gif():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        datai = decode_gif(txt)
        txt = encode_gif(datai)
        ldata = decode_gif(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        txt = load_bytes(filename)
        datai = decode_gif(txt, which="all")
        txt = encode_gif(datai)
        ldata = decode_gif(txt, which="all")
        assert _video_distance(ldata['video'], data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_gif():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif(f, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif(f, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        with open(filename, "rb") as f:
            ldata = read_gif(f, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_write_gif():
    filename = FILENAMES_GIF.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        with open(filename2, "rb") as f:
            datai = read_gif(f)
        with open(filename, "wb") as f:
            write_gif(f, datai)
        with open(filename, "rb") as f:
            ldata = read_gif(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        with open(filename2, "rb") as f:
            datai = read_gif(f, which="all")
        with open(filename, "wb") as f:
            write_gif(f, datai)
        with open(filename, "rb") as f:
            ldata = read_gif(f, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_load_gif():
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif(filename, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif(filename, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "gif"
    for i in range(len(data)):
        filename = FILENAMES_GIF.format(i)
        ldata = load_gif(filename, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "gif"


def test_save_gif():
    filename = FILENAMES_GIF.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        datai = load_gif(filename2)
        save_gif(filename, datai)
        ldata = load_gif(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_GIF.format(i)
        datai = load_gif(filename2, which="all")
        save_gif(filename, datai)
        ldata = load_gif(filename, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_help_gif():
    help_gif()
