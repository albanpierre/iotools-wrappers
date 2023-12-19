
import os
import numpy as np
import PIL.Image as PilImage

from iotools.bytesio import load_bytes
from iotools.simplified.jpgio_using_imageio import (
    decode_jpg_using_imageio, encode_jpg_using_imageio,
    read_jpg_using_imageio, write_jpg_using_imageio,
    load_jpg_using_imageio, save_jpg_using_imageio
)
from iotools.simplified.jpgio_using_skimage import (
    decode_jpg_using_skimage, encode_jpg_using_skimage,
    read_jpg_using_skimage, write_jpg_using_skimage,
    load_jpg_using_skimage, save_jpg_using_skimage
)
from iotools.simplified.jpgio_using_pil import (
    decode_jpg_using_pil, encode_jpg_using_pil,
    read_jpg_using_pil, write_jpg_using_pil,
    load_jpg_using_pil, save_jpg_using_pil
)
from iotools.simplified.jpgio_using_cv2 import (
    decode_jpg_using_cv2, encode_jpg_using_cv2,
    read_jpg_using_cv2, write_jpg_using_cv2,
    load_jpg_using_cv2, save_jpg_using_cv2
)
from iotools.simplified.jpgio import (
    decode_jpg, encode_jpg, read_jpg, write_jpg, load_jpg, save_jpg
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_JPG = os.path.join(TEST_DATA_PATH, "image", "example_jpg_{}.jpg")

image = {}
image[0] = np.asarray([[[255, 255, 255, 255],
                        [255, 196, 128, 255],
                        [255, 128, 0, 255],
                        [255, 64, 128, 255],
                        [255, 0, 255, 255]],
                       [[128, 255, 255, 255],
                        [128, 128, 128, 255],
                        [128, 127, 1, 255],
                        [128, 63, 127, 255],
                        [128, 0, 255, 255]],
                       [[64, 255, 255, 255],
                        [64, 194, 126, 255],
                        [64, 126, 10, 255],
                        [64, 62, 130, 255],
                        [64, 2, 250, 255]],
                       [[0, 255, 33, 255],
                        [0, 127, 127, 255],
                        [0, 0, 0, 255],
                        [0, 128, 128, 255],
                        [0, 255, 255, 255]]], dtype=np.uint8)


data = {}
data[0] = image[0][:, :, :3].copy()


def _format_image(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis].repeat(4, axis=2)
        img[:, :, 3] = 255
    if img.shape[2] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255)], axis=2)
    return img.astype(np.uint8)


def _image_distance(img1, img2):
    img1 = _format_image(img1)
    img2 = _format_image(img2)
    if img1.shape != img2.shape:
        return 256
    return np.mean(np.abs(img1.astype(int) - img2.astype(int)))


# +---------------+
# | Using imageio |
# +---------------+


def test_help_decode_jpg_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        txt = load_bytes(filename)
        ldata = decode_jpg_using_imageio(txt)
        assert _image_distance(ldata, data[i]) < 32, "Error while decoding jpg image nbr {}".format(i)


def test_help_encode_jpg_using_imageio():
    for kwargs in [{}]:
        for i in range(len(data)):
            txt = encode_jpg_using_imageio(data[i], **kwargs)
            ldata = decode_jpg_using_imageio(txt)
            assert _image_distance(ldata, data[i]) < 32, "Error while encoding jpg image nbr {}".format(i)


def test_help_read_jpg_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        with open(filename, "rb") as f:
            ldata = read_jpg_using_imageio(f)
        assert _image_distance(ldata, data[i]) < 32, "Error while reading jpg image nbr {}".format(i)


def test_help_write_jpg_using_imageio():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_jpg_using_imageio(f, data[i], **kwargs)
            with open(filename, "rb") as f:
                ldata = read_jpg_using_imageio(f)
            assert _image_distance(ldata, data[i]) < 32, "Error while writing jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_load_jpg_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        ldata = load_jpg_using_imageio(filename)
        assert _image_distance(ldata, data[i]) < 32, "Error while loading jpg image nbr {}".format(i)


def test_help_save_jpg_using_imageio():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            save_jpg_using_imageio(filename, data[i], **kwargs)
            ldata = load_jpg_using_imageio(filename)
            assert _image_distance(ldata, data[i]) < 32, "Error while saving jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_jpg_using_imageio():
    save_jpg_using_imageio("filename.jpg", data[0])
    import iotools.examples.examples_jpg_using_imageio  # noqa
    os.remove("filename.jpg")


# +---------------+
# | Using skimage |
# +---------------+


def test_help_decode_jpg_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        txt = load_bytes(filename)
        ldata = decode_jpg_using_skimage(txt)
        assert _image_distance(ldata, data[i]) < 32, "Error while decoding jpg image nbr {}".format(i)


def test_help_encode_jpg_using_skimage():
    for kwargs in [{}]:
        for i in range(len(data)):
            txt = encode_jpg_using_skimage(data[i], **kwargs)
            ldata = decode_jpg_using_skimage(txt)
            assert _image_distance(ldata, data[i]) < 32, "Error while encoding jpg image nbr {}".format(i)


def test_help_read_jpg_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        with open(filename, "rb") as f:
            ldata = read_jpg_using_skimage(f)
        assert _image_distance(ldata, data[i]) < 32, "Error while reading jpg image nbr {}".format(i)


def test_help_write_jpg_using_skimage():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_jpg_using_skimage(f, data[i], **kwargs)
            with open(filename, "rb") as f:
                ldata = read_jpg_using_skimage(f)
            assert _image_distance(ldata, data[i]) < 32, "Error while writing jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_load_jpg_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        ldata = load_jpg_using_skimage(filename)
        assert _image_distance(ldata, data[i]) < 32, "Error while loading jpg image nbr {}".format(i)


def test_help_save_jpg_using_skimage():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            save_jpg_using_skimage(filename, data[i], **kwargs)
            ldata = load_jpg_using_skimage(filename)
            assert _image_distance(ldata, data[i]) < 32, "Error while saving jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_jpg_using_skimage():
    save_jpg_using_skimage("filename.jpg", data[0])
    import iotools.examples.examples_jpg_using_skimage  # noqa
    os.remove("filename.jpg")


# +-----------+
# | Using PIL |
# +-----------+


def test_help_decode_jpg_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        txt = load_bytes(filename)
        ldata = decode_jpg_using_pil(txt)
        assert _image_distance(ldata, data[i]) < 32, "Error while decoding jpg image nbr {}".format(i)


def test_help_encode_jpg_using_pil():
    for kwargs in [{}]:
        for i in range(len(data)):
            if (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            txt = encode_jpg_using_pil(PilImage.fromarray(data[i]), **kwargs)
            ldata = decode_jpg_using_pil(txt)
            assert _image_distance(ldata, data[i]) < 32, "Error while encoding jpg image nbr {}".format(i)


def test_help_read_jpg_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        with open(filename, "rb") as f:
            ldata = read_jpg_using_pil(f)
        assert _image_distance(ldata, data[i]) < 32, "Error while reading jpg image nbr {}".format(i)


def test_help_write_jpg_using_pil():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            if (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            with open(filename, "wb") as f:
                write_jpg_using_pil(f, PilImage.fromarray(data[i]), **kwargs)
            with open(filename, "rb") as f:
                ldata = read_jpg_using_pil(f)
            assert _image_distance(ldata, data[i]) < 32, "Error while writing jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_load_jpg_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        ldata = load_jpg_using_pil(filename)
        assert _image_distance(ldata, data[i]) < 32, "Error while loading jpg image nbr {}".format(i)


def test_help_save_jpg_using_pil():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            if (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            save_jpg_using_pil(filename, PilImage.fromarray(data[i]), **kwargs)
            ldata = load_jpg_using_pil(filename)
            assert _image_distance(ldata, data[i]) < 32, "Error while saving jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_jpg_using_pil():
    save_jpg_using_pil("filename.jpg", PilImage.fromarray(data[0]))
    import iotools.examples.examples_jpg_using_pil  # noqa
    os.remove("filename.jpg")


# +-----------+
# | Using cv2 |
# +-----------+


def test_help_decode_jpg_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        txt = load_bytes(filename)
        ldata = decode_jpg_using_cv2(txt)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) < 32, "Error while decoding jpg image nbr {}".format(i)


def test_help_encode_jpg_using_cv2():
    for kwargs in [{}]:
        for i in range(len(data)):
            txt = encode_jpg_using_cv2(data[i], **kwargs)
            ldata = decode_jpg_using_cv2(txt)
            assert _image_distance(ldata, data[i]) < 32, "Error while encoding jpg image nbr {}".format(i)


def test_help_read_jpg_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        with open(filename, "rb") as f:
            ldata = read_jpg_using_cv2(f)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) < 32, "Error while reading jpg image nbr {}".format(i)


def test_help_write_jpg_using_cv2():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_jpg_using_cv2(f, data[i], **kwargs)
            with open(filename, "rb") as f:
                ldata = read_jpg_using_cv2(f)
            assert _image_distance(ldata, data[i]) < 32, "Error while writing jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_load_jpg_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        ldata = load_jpg_using_cv2(filename)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) < 32, "Error while loading jpg image nbr {}".format(i)


def test_help_save_jpg_using_cv2():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            save_jpg_using_cv2(filename, data[i], **kwargs)
            ldata = load_jpg_using_cv2(filename)
            assert _image_distance(ldata, data[i]) < 32, "Error while saving jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_jpg_using_cv2():
    save_jpg_using_cv2("filename.jpg", data[0])
    import iotools.examples.examples_jpg_using_cv2  # noqa
    os.remove("filename.jpg")


# +------------------+
# | Default behavior |
# +------------------+


def test_help_decode_jpg():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        txt = load_bytes(filename)
        ldata = decode_jpg(txt)
        assert _image_distance(ldata, data[i]) < 32, "Error while decoding jpg image nbr {}".format(i)


def test_help_encode_jpg():
    for kwargs in [{}]:
        for i in range(len(data)):
            txt = encode_jpg(data[i], **kwargs)
            ldata = decode_jpg(txt)
            assert _image_distance(ldata, data[i]) < 32, "Error while encoding jpg image nbr {}".format(i)


def test_help_read_jpg():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        with open(filename, "rb") as f:
            ldata = read_jpg(f)
        assert _image_distance(ldata, data[i]) < 32, "Error while reading jpg image nbr {}".format(i)


def test_help_write_jpg():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_jpg(f, data[i], **kwargs)
            with open(filename, "rb") as f:
                ldata = read_jpg(f)
            assert _image_distance(ldata, data[i]) < 32, "Error while writing jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_load_jpg():
    for i in range(len(data)):
        filename = FILENAMES_JPG.format(i)
        ldata = load_jpg(filename)
        assert _image_distance(ldata, data[i]) < 32, "Error while loading jpg image nbr {}".format(i)


def test_help_save_jpg():
    filename = FILENAMES_JPG.format("write")
    for kwargs in [{}]:
        for i in range(len(data)):
            save_jpg(filename, data[i], **kwargs)
            ldata = load_jpg(filename)
            assert _image_distance(ldata, data[i]) < 32, "Error while saving jpg image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_jpg():
    save_jpg("filename.jpg", data[0])
    import iotools.examples.examples_jpg  # noqa
    os.remove("filename.jpg")
