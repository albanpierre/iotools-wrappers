
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.simplified.videoio_using_imageio import (
    decode_video_using_imageio, encode_video_using_imageio,
    read_video_using_imageio, write_video_using_imageio,
    load_video_using_imageio, save_video_using_imageio,
)
from iotools.simplified.videoio_using_cv2 import (
    decode_video_using_cv2, encode_video_using_cv2,
    read_video_using_cv2, write_video_using_cv2,
    load_video_using_cv2, save_video_using_cv2,
)
from iotools.simplified.videoio import decode_video, encode_video, read_video, write_video, load_video, save_video


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_MP4 = os.path.join(TEST_DATA_PATH, "video", "example_mp4_{}.mp4")
FILENAMES_GIF = os.path.join(TEST_DATA_PATH, "video", "example_gif_{}.gif")
FILENAMES_AVI = os.path.join(TEST_DATA_PATH, "video", "example_avi_{}.avi")

data = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(6)]
for i in range(6):
    for x in range(3):
        for y in range(3):
            data[i][(i // 3) * 16 + x, (i % 3) * 16 + y, :] = [55, 155, 255]

data_mp4 = {}
data_mp4[0] = data
data_mp4[1] = data
data_gif = {}
data_gif[0] = data
data_avi = {}
data_avi[0] = data


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


read_list_of_args = list(zip(
    [FILENAMES_MP4, FILENAMES_GIF, FILENAMES_AVI],
    [data_mp4, data_gif, data_avi],
    ["mp4", "gif", "avi"],
))

write_list_of_args = list(zip(
    [FILENAMES_MP4, FILENAMES_GIF, FILENAMES_AVI],
    [data_mp4, data_gif, data_avi],
    ["mp4", "gif", "avi"],
    ["mp4v", None, "MJPG"],
))


# +---------------+
# | Using imageio |
# +---------------+


def test_decode_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_imageio(txt, format=ext)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_encode_video_using_imageio():
    for filenames_video, data, ext, codec in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video_using_imageio(txt, format=ext)
            txt = encode_video_using_imageio(datai, format=ext)
            ldata = decode_video_using_imageio(txt, format=ext)
            assert _video_distance(ldata, data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, format=ext)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_write_video_using_imageio():
    for filenames_video, data, ext, codec in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video_using_imageio(f, format=ext)
            with open(filename, "wb") as f:
                write_video_using_imageio(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, format=ext)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)


def test_load_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_imageio(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_save_video_using_imageio():
    for filenames_video, data, ext, codec in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video_using_imageio(filename2)
            save_video_using_imageio(filename, datai, format=ext)
            ldata = load_video_using_imageio(filename)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)


def test_help_examples_video_using_imageio():
    save_video_using_imageio("filename.mp4", data_mp4[0])
    import iotools.examples.examples_video_using_imageio  # noqa
    os.remove("filename.mp4")


# +-----------+
# | Using cv2 |
# +-----------+


def test_decode_video_using_cv2():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_cv2(txt)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_encode_video_using_cv2():
    for filenames_video, data, ext, codec in write_list_of_args:
        if codec is not None:
            for i in range(len(data)):
                filename = filenames_video.format(i)
                txt = load_bytes(filename)
                datai = decode_video_using_cv2(txt)
                txt = encode_video_using_cv2(datai, format=ext, codec=codec)
                ldata = decode_video_using_cv2(txt)
                assert _video_distance(ldata, data[i]) < 16
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_video_using_cv2():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_cv2(f)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_write_video_using_cv2():
    for filenames_video, data, ext, codec in write_list_of_args:
        if codec is not None:
            filename = filenames_video.format("write")
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                with open(filename2, "rb") as f:
                    datai = read_video_using_cv2(f)
                with open(filename, "wb") as f:
                    write_video_using_cv2(f, datai, format=ext, codec=codec)
                with open(filename, "rb") as f:
                    ldata = read_video_using_cv2(f)
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5
            os.remove(filename)


def test_load_video_using_cv2():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_cv2(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_save_video_using_cv2():
    for filenames_video, data, ext, codec in write_list_of_args:
        if codec is not None:
            filename = filenames_video.format("write")
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                datai = load_video_using_cv2(filename2)
                save_video_using_cv2(filename, datai, codec=codec)
                ldata = load_video_using_cv2(filename)
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5
            os.remove(filename)


def test_help_examples_video_using_cv2():
    save_video_using_cv2("filename.mp4", data_mp4[0])
    import iotools.examples.examples_video_using_cv2  # noqa
    os.remove("filename.mp4")


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video(txt, format=ext)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_encode_video():
    for filenames_video, data, ext, codec in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video(txt, format=ext)
            txt = encode_video(datai, format=ext)
            ldata = decode_video(txt, format=ext)
            assert _video_distance(ldata, data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5


def test_read_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video(f, format=ext)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_write_video():
    for filenames_video, data, ext, codec in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video(f, format=ext)
            with open(filename, "wb") as f:
                write_video(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video(f, format=ext)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)


def test_load_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16


def test_save_video():
    for filenames_video, data, ext, codec in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video(filename2)
            save_video(filename, datai, format=ext)
            ldata = load_video(filename)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)


def test_help_examples_video():
    save_video("filename.mp4", data_mp4[0])
    import iotools.examples.examples_video  # noqa
    os.remove("filename.mp4")
