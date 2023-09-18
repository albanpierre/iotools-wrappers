
import os
import numpy as np

from iotools.bytesio import load_bytes
from simplified_iotools.aviio_using_imageio import (
    decode_avi_using_imageio, encode_avi_using_imageio,
    read_avi_using_imageio, write_avi_using_imageio,
    load_avi_using_imageio, save_avi_using_imageio,
)
from simplified_iotools.aviio_using_cv2 import (
    decode_avi_using_cv2, encode_avi_using_cv2,
    read_avi_using_cv2, write_avi_using_cv2,
    load_avi_using_cv2, save_avi_using_cv2,
)
from simplified_iotools.aviio import decode_avi, encode_avi, read_avi, write_avi, load_avi, save_avi


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_AVI = os.path.join(TEST_DATA_PATH, "video", "example_avi_{}.avi")

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


def test_decode_avi_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        ldata = decode_avi_using_imageio(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_encode_avi_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        datai = decode_avi_using_imageio(txt)
        txt = encode_avi_using_imageio(datai)
        ldata = decode_avi_using_imageio(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_avi_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        with open(filename, "rb") as f:
            ldata = read_avi_using_imageio(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_write_avi_using_imageio():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        with open(filename2, "rb") as f:
            datai = read_avi_using_imageio(f)
        with open(filename, "wb") as f:
            write_avi_using_imageio(f, datai)
        with open(filename, "rb") as f:
            ldata = read_avi_using_imageio(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_load_avi_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        ldata = load_avi_using_imageio(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_save_avi_using_imageio():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        datai = load_avi_using_imageio(filename2)
        save_avi_using_imageio(filename, datai)
        ldata = load_avi_using_imageio(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_help_examples_avi_using_imageio():
    save_avi_using_imageio("filename.avi", data[0])
    import examples.examples_avi_using_imageio  # noqa
    os.remove("filename.avi")


# +-----------+
# | Using cv2 |
# +-----------+


def test_decode_avi_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        ldata = decode_avi_using_cv2(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_encode_avi_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        datai = decode_avi_using_cv2(txt)
        txt = encode_avi_using_cv2(datai)
        ldata = decode_avi_using_cv2(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_avi_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        with open(filename, "rb") as f:
            ldata = read_avi_using_cv2(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_write_avi_using_cv2():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        with open(filename2, "rb") as f:
            datai = read_avi_using_cv2(f)
        with open(filename, "wb") as f:
            write_avi_using_cv2(f, datai)
        with open(filename, "rb") as f:
            ldata = read_avi_using_cv2(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_load_avi_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        ldata = load_avi_using_cv2(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_save_avi_using_cv2():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        datai = load_avi_using_cv2(filename2)
        save_avi_using_cv2(filename, datai)
        ldata = load_avi_using_cv2(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_help_examples_avi_using_cv2():
    save_avi_using_cv2("filename.avi", data[0])
    import examples.examples_avi_using_cv2  # noqa
    os.remove("filename.avi")


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_avi():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        ldata = decode_avi(txt)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_encode_avi():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        txt = load_bytes(filename)
        datai = decode_avi(txt)
        txt = encode_avi(datai)
        ldata = decode_avi(txt)
        assert _video_distance(ldata, data[i]) < 16
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_avi():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        with open(filename, "rb") as f:
            ldata = read_avi(f)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_write_avi():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        with open(filename2, "rb") as f:
            datai = read_avi(f)
        with open(filename, "wb") as f:
            write_avi(f, datai)
        with open(filename, "rb") as f:
            ldata = read_avi(f)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_load_avi():
    for i in range(len(data)):
        filename = FILENAMES_AVI.format(i)
        ldata = load_avi(filename)
        assert not isinstance(ldata, dict)
        assert _video_distance(ldata, data[i]) < 16


def test_save_avi():
    filename = FILENAMES_AVI.format("write")
    for i in range(len(data)):
        filename2 = FILENAMES_AVI.format(i)
        datai = load_avi(filename2)
        save_avi(filename, datai)
        ldata = load_avi(filename)
        assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    os.remove(filename)


def test_help_examples_avi():
    save_avi("filename.avi", data[0])
    import examples.examples_avi  # noqa
    os.remove("filename.avi")
