
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.videoio import (
    DEFAULT_PROPERTIES_CV2,
    decode_video_using_imageio, encode_video_using_imageio,
    read_video_using_imageio, write_video_using_imageio,
    load_video_using_imageio, save_video_using_imageio, help_video_using_imageio,
    decode_video_using_cv2, encode_video_using_cv2,
    read_video_using_cv2, write_video_using_cv2,
    load_video_using_cv2, save_video_using_cv2, help_video_using_cv2,
    decode_video, encode_video, read_video, write_video, load_video, save_video, help_video
)


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
))


# +---------------+
# | Using imageio |
# +---------------+


def test_decode_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_imageio(txt)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_imageio(txt, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_imageio(txt, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_imageio(txt, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_encode_video_using_imageio():
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video_using_imageio(txt)
            txt = encode_video_using_imageio(datai, format=ext)
            ldata = decode_video_using_imageio(txt)
            assert _video_distance(ldata, data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video_using_imageio(txt, which="all")
            txt = encode_video_using_imageio(datai, format=ext)
            ldata = decode_video_using_imageio(txt, which="all")
            assert _video_distance(ldata['video'], data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_write_video_using_imageio():
    for filenames_video, data, ext in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video_using_imageio(f)
            with open(filename, "wb") as f:
                write_video_using_imageio(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video_using_imageio(f, which="all")
            with open(filename, "wb") as f:
                write_video_using_imageio(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video_using_imageio(f, which="all")
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
        os.remove(filename)


def test_load_video_using_imageio():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_imageio(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_imageio(filename, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_imageio(filename, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_imageio(filename, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_save_video_using_imageio():
    for filenames_video, data, ext in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video_using_imageio(filename2)
            save_video_using_imageio(filename, datai)
            ldata = load_video_using_imageio(filename)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video_using_imageio(filename2, which="all")
            save_video_using_imageio(filename, datai)
            ldata = load_video_using_imageio(filename, which="all")
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
        os.remove(filename)


def test_help_video_using_imageio():
    help_video_using_imageio()


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
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_cv2(txt, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_cv2(txt, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video_using_cv2(txt, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_encode_video_using_cv2():
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            for i in range(len(data)):
                filename = filenames_video.format(i)
                txt = load_bytes(filename)
                datai = decode_video_using_cv2(txt)
                txt = encode_video_using_cv2(datai, format=ext)
                ldata = decode_video_using_cv2(txt)
                assert _video_distance(ldata, data[i]) < 16
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            for i in range(len(data)):
                filename = filenames_video.format(i)
                txt = load_bytes(filename)
                datai = decode_video_using_cv2(txt, which="all")
                del datai["properties"]["codec"]
                # del datai["properties"]["format"]
                txt = encode_video_using_cv2(datai, format=ext)
                ldata = decode_video_using_cv2(txt, which="all")
                assert _video_distance(ldata['video'], data[i]) < 16
                assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_video_using_cv2():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_cv2(f)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_cv2(f, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_cv2(f, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video_using_cv2(f, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_write_video_using_cv2():
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            filename = filenames_video.format("write")
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                with open(filename2, "rb") as f:
                    datai = read_video_using_cv2(f)
                with open(filename, "wb") as f:
                    write_video_using_cv2(f, datai, format=ext)
                with open(filename, "rb") as f:
                    ldata = read_video_using_cv2(f)
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5
            os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                with open(filename2, "rb") as f:
                    datai = read_video_using_cv2(f, which="all")
                del datai["properties"]["codec"]
                with open(filename, "wb") as f:
                    write_video_using_cv2(f, datai, format=ext)
                with open(filename, "rb") as f:
                    ldata = read_video_using_cv2(f, which="all")
                assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
            os.remove(filename)


def test_load_video_using_cv2():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_cv2(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_cv2(filename, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_cv2(filename, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video_using_cv2(filename, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_save_video_using_cv2():
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            filename = filenames_video.format("write")
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                datai = load_video_using_cv2(filename2)
                save_video_using_cv2(filename, datai)
                ldata = load_video_using_cv2(filename)
                assert np.max(_video_distance_per_frame(ldata, datai)) < 5
            os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        if DEFAULT_PROPERTIES_CV2[ext]["codec"] is not None:
            for i in range(len(data)):
                filename2 = filenames_video.format(i)
                datai = load_video_using_cv2(filename2, which="all")
                del datai["properties"]["codec"]
                save_video_using_cv2(filename, datai)
                ldata = load_video_using_cv2(filename, which="all")
                assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
            os.remove(filename)


def test_help_video_using_cv2():
    help_video_using_cv2()


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video(txt)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video(txt, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video(txt, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            ldata = decode_video(txt, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_encode_video():
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video(txt)
            txt = encode_video(datai, format=ext)
            ldata = decode_video(txt)
            assert _video_distance(ldata, data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            txt = load_bytes(filename)
            datai = decode_video(txt, which="all")
            txt = encode_video(datai, format=ext)
            ldata = decode_video(txt, which="all")
            assert _video_distance(ldata['video'], data[i]) < 16
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5


def test_read_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video(f)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video(f, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video(f, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            with open(filename, "rb") as f:
                ldata = read_video(f, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_write_video():
    for filenames_video, data, ext in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video(f)
            with open(filename, "wb") as f:
                write_video(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video(f)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            with open(filename2, "rb") as f:
                datai = read_video(f, which="all")
            with open(filename, "wb") as f:
                write_video(f, datai, format=ext)
            with open(filename, "rb") as f:
                ldata = read_video(f, which="all")
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
        os.remove(filename)


def test_load_video():
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video(filename)
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video(filename, which="video")
            assert not isinstance(ldata, dict)
            assert _video_distance(ldata, data[i]) < 16
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video(filename, which="properties")
            assert isinstance(ldata, dict)
            assert ldata["format"] == ext
    for filenames_video, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_video.format(i)
            ldata = load_video(filename, which="all")
            assert isinstance(ldata, dict)
            assert _video_distance(ldata["video"], data[i]) < 16
            assert ldata["properties"]["format"] == ext


def test_save_video():
    for filenames_video, data, ext in write_list_of_args:
        filename = filenames_video.format("write")
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video(filename2)
            save_video(filename, datai)
            ldata = load_video(filename)
            assert np.max(_video_distance_per_frame(ldata, datai)) < 5
        os.remove(filename)
    for filenames_video, data, ext in write_list_of_args:
        for i in range(len(data)):
            filename2 = filenames_video.format(i)
            datai = load_video(filename2, which="all")
            save_video(filename, datai)
            ldata = load_video(filename, which="all")
            assert np.max(_video_distance_per_frame(ldata['video'], datai['video'])) < 5
        os.remove(filename)


def test_help_video():
    help_video()
