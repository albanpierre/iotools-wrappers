
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.mp4io import (
    decode_mp4_using_imageio, encode_mp4_using_imageio,
    read_mp4_using_imageio, write_mp4_using_imageio,
    load_mp4_using_imageio, save_mp4_using_imageio, help_mp4_using_imageio,
    decode_mp4_using_cv2, encode_mp4_using_cv2,
    read_mp4_using_cv2, write_mp4_using_cv2,
    load_mp4_using_cv2, save_mp4_using_cv2, help_mp4_using_cv2,
    decode_mp4, encode_mp4, read_mp4, write_mp4, load_mp4, save_mp4, help_mp4
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_MP4 = os.path.join(TEST_DATA_PATH, "video", "example_mp4_{}.mp4")

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


def test_decode_mp4_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_imageio(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_imageio(txt, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_imageio(txt, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_imageio(txt, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_encode_mp4_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4_using_imageio(txt)
        txt = encode_mp4_using_imageio(datai)
        ldata = decode_mp4_using_imageio(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4_using_imageio(txt, which="all")
        txt = encode_mp4_using_imageio(datai)
        ldata = decode_mp4_using_imageio(txt, which="all")
        assert _video_distance(ldata['video'], data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_mp4_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_write_mp4_using_imageio():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4_using_imageio(f)
        with open(filename, "wb") as f:
            write_mp4_using_imageio(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4_using_imageio(f, which="all")
        with open(filename, "wb") as f:
            write_mp4_using_imageio(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_imageio(f, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_load_mp4_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_imageio(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_imageio(filename, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_imageio(filename, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_imageio(filename, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_save_mp4_using_imageio():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4_using_imageio(filename2)
        save_mp4_using_imageio(filename, datai)
        ldata = load_mp4_using_imageio(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4_using_imageio(filename2, which="all")
        save_mp4_using_imageio(filename, datai)
        ldata = load_mp4_using_imageio(filename, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_help_mp4_using_imageio():
    help_mp4_using_imageio()


# +-----------+
# | Using cv2 |
# +-----------+


def test_decode_mp4_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_cv2(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_cv2(txt, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_cv2(txt, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4_using_cv2(txt, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_encode_mp4_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4_using_cv2(txt)
        txt = encode_mp4_using_cv2(datai)
        ldata = decode_mp4_using_cv2(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4_using_cv2(txt, which="all")
        del datai["properties"]["codec"]
        # del datai["properties"]["format"]
        txt = encode_mp4_using_cv2(datai)
        ldata = decode_mp4_using_cv2(txt, which="all")
        assert _video_distance(ldata['video'], data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_mp4_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_write_mp4_using_cv2():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4_using_cv2(f)
        with open(filename, "wb") as f:
            write_mp4_using_cv2(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4_using_cv2(f, which="all")
        del datai["properties"]["codec"]
        with open(filename, "wb") as f:
            write_mp4_using_cv2(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4_using_cv2(f, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_load_mp4_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_cv2(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_cv2(filename, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_cv2(filename, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4_using_cv2(filename, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_save_mp4_using_cv2():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4_using_cv2(filename2)
        save_mp4_using_cv2(filename, datai)
        ldata = load_mp4_using_cv2(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4_using_cv2(filename2, which="all")
        del datai["properties"]["codec"]
        save_mp4_using_cv2(filename, datai)
        ldata = load_mp4_using_cv2(filename, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_help_mp4_using_cv2():
    help_mp4_using_cv2()


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_mp4():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4(txt, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4(txt, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        ldata = decode_mp4(txt, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_encode_mp4():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4(txt)
        txt = encode_mp4(datai)
        ldata = decode_mp4(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        txt = load_bytes(filename)
        datai = decode_mp4(txt, which="all")
        txt = encode_mp4(datai)
        ldata = decode_mp4(txt, which="all")
        assert _video_distance(ldata['video'], data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_mp4():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4(f, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4(f, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        with open(filename, "rb") as f:
            ldata = read_mp4(f, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_write_mp4():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4(f)
        with open(filename, "wb") as f:
            write_mp4(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        with open(filename2, "rb") as f:
            datai = read_mp4(f, which="all")
        with open(filename, "wb") as f:
            write_mp4(f, datai)
        with open(filename, "rb") as f:
            ldata = read_mp4(f, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_load_mp4():
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4(filename, which="video")
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4(filename, which="properties")
        assert isinstance(ldata, dict)
        assert ldata["format"] == "mp4"
    for i in range(len(data)):
        filename = FILENAMES_MP4.format(i)
        ldata = load_mp4(filename, which="all")
        assert isinstance(ldata, dict)
        assert _video_distance(ldata["video"], data[i]) < 16
        assert ldata["properties"]["format"] == "mp4"


def test_save_mp4():
    filename = FILENAMES_MP4.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4(filename2)
        save_mp4(filename, datai)
        ldata = load_mp4(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)
    for i in range(len(data)):
        filename2 = FILENAMES_MP4.format(i)
        datai = load_mp4(filename2, which="all")
        save_mp4(filename, datai)
        ldata = load_mp4(filename, which="all")
        assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
    os.remove(filename)


def test_help_mp4():
    help_mp4()
