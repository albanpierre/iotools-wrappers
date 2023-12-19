
import os
import numpy as np
import PIL.Image as PilImage

from iotools.bytesio import load_bytes
from iotools.simplified.pngio_using_imageio import (
    decode_png_using_imageio, encode_png_using_imageio,
    read_png_using_imageio, write_png_using_imageio,
    load_png_using_imageio, save_png_using_imageio
)
from iotools.simplified.pngio_using_skimage import (
    decode_png_using_skimage, encode_png_using_skimage,
    read_png_using_skimage, write_png_using_skimage,
    load_png_using_skimage, save_png_using_skimage
)
from iotools.simplified.pngio_using_pil import (
    decode_png_using_pil, encode_png_using_pil,
    read_png_using_pil, write_png_using_pil,
    load_png_using_pil, save_png_using_pil
)
from iotools.simplified.pngio_using_cv2 import (
    decode_png_using_cv2, encode_png_using_cv2,
    read_png_using_cv2, write_png_using_cv2,
    load_png_using_cv2, save_png_using_cv2
)
from iotools.simplified.pngio import (
    decode_png, encode_png, read_png, write_png, load_png, save_png
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_PNG = os.path.join(TEST_DATA_PATH, "image", "example_png_{}.png")

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

image[1] = image[0][:, :, :3].copy()
image[2] = image[0][:, :, 1].copy()
image[3] = image[0].copy()
image[3][:, :, 0] = image[2]
image[3][:, :, 1] = image[2]
image[3][:, :, 2] = image[2]

data = {}
for i in range(4):
    data[i] = image[i].copy()


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


def test_help_decode_png_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        txt = load_bytes(filename)
        ldata = decode_png_using_imageio(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while decoding png image nbr {}".format(i)


def test_help_encode_png_using_imageio():
    for i in range(len(data)):
        txt = encode_png_using_imageio(data[i])
        ldata = decode_png_using_imageio(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while encoding png image nbr {}".format(i)


def test_help_read_png_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        with open(filename, "rb") as f:
            ldata = read_png_using_imageio(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while reading png image nbr {}".format(i)


def test_help_write_png_using_imageio():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_png_using_imageio(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_png_using_imageio(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while writing png image nbr {}".format(i)
    os.remove(filename)


def test_help_load_png_using_imageio():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        ldata = load_png_using_imageio(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while loading png image nbr {}".format(i)


def test_help_save_png_using_imageio():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        save_png_using_imageio(filename, data[i])
        ldata = load_png_using_imageio(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while saving png image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_png_using_imageio():
    save_png_using_imageio("filename.png", data[0])
    import iotools.examples.examples_png_using_imageio  # noqa
    os.remove("filename.png")


# +---------------+
# | Using skimage |
# +---------------+


def test_help_decode_png_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        txt = load_bytes(filename)
        ldata = decode_png_using_skimage(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while decoding png image nbr {}".format(i)


def test_help_encode_png_using_skimage():
    for i in range(len(data)):
        txt = encode_png_using_skimage(data[i])
        ldata = decode_png_using_skimage(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while encoding png image nbr {}".format(i)


def test_help_read_png_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        with open(filename, "rb") as f:
            ldata = read_png_using_skimage(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while reading png image nbr {}".format(i)


def test_help_write_png_using_skimage():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_png_using_skimage(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_png_using_skimage(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while writing png image nbr {}".format(i)
    os.remove(filename)


def test_help_load_png_using_skimage():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        ldata = load_png_using_skimage(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while loading png image nbr {}".format(i)


def test_help_save_png_using_skimage():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        save_png_using_skimage(filename, data[i])
        ldata = load_png_using_skimage(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while saving png image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_png_using_skimage():
    save_png_using_skimage("filename.png", data[0])
    import iotools.examples.examples_png_using_skimage  # noqa
    os.remove("filename.png")


# +-----------+
# | Using PIL |
# +-----------+


def test_help_decode_png_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        txt = load_bytes(filename)
        ldata = decode_png_using_pil(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while decoding png image nbr {}".format(i)


def test_help_encode_png_using_pil():
    for i in range(len(data)):
        if (data[i].ndim == 3) and (data[i].shape[2] == 4):
            continue  # do not test RGBA png images since PIL will raise an error in that case
        txt = encode_png_using_pil(PilImage.fromarray(data[i]))
        ldata = decode_png_using_pil(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while encoding png image nbr {}".format(i)


def test_help_read_png_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        with open(filename, "rb") as f:
            ldata = read_png_using_pil(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while reading png image nbr {}".format(i)


def test_help_write_png_using_pil():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        if (data[i].ndim == 3) and (data[i].shape[2] == 4):
            continue  # do not test RGBA png images since PIL will raise an error in that case
        with open(filename, "wb") as f:
            write_png_using_pil(f, PilImage.fromarray(data[i]))
        with open(filename, "rb") as f:
            ldata = read_png_using_pil(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while writing png image nbr {}".format(i)
    os.remove(filename)


def test_help_load_png_using_pil():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        ldata = load_png_using_pil(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while loading png image nbr {}".format(i)


def test_help_save_png_using_pil():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        if (data[i].ndim == 3) and (data[i].shape[2] == 4):
            continue  # do not test RGBA png images since PIL will raise an error in that case
        save_png_using_pil(filename, PilImage.fromarray(data[i]))
        ldata = load_png_using_pil(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while saving png image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_png_using_pil():
    save_png_using_pil("filename.png", PilImage.fromarray(data[0]))
    import iotools.examples.examples_png_using_pil  # noqa
    os.remove("filename.png")


# +-----------+
# | Using cv2 |
# +-----------+


def test_help_decode_png_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        txt = load_bytes(filename)
        ldata = decode_png_using_cv2(txt)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) == 0, "Error while decoding png image nbr {}".format(i)


def test_help_encode_png_using_cv2():
    for i in range(len(data)):
        txt = encode_png_using_cv2(data[i])
        ldata = decode_png_using_cv2(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while encoding png image nbr {}".format(i)


def test_help_read_png_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        with open(filename, "rb") as f:
            ldata = read_png_using_cv2(f)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) == 0, "Error while reading png image nbr {}".format(i)


def test_help_write_png_using_cv2():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_png_using_cv2(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_png_using_cv2(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while writing png image nbr {}".format(i)
    os.remove(filename)


def test_help_load_png_using_cv2():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        ldata = load_png_using_cv2(filename)
        ldata = _format_image(ldata)
        ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
        assert _image_distance(ldata, data[i]) == 0, "Error while loading png image nbr {}".format(i)


def test_help_save_png_using_cv2():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        save_png_using_cv2(filename, data[i])
        ldata = load_png_using_cv2(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while saving png image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_png_using_cv2():
    save_png_using_cv2("filename.png", data[0])
    import iotools.examples.examples_png_using_cv2  # noqa
    os.remove("filename.png")


# +------------------+
# | Default behavior |
# +------------------+


def test_help_decode_png():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        txt = load_bytes(filename)
        ldata = decode_png(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while decoding png image nbr {}".format(i)


def test_help_encode_png():
    for i in range(len(data)):
        txt = encode_png(data[i])
        ldata = decode_png(txt)
        assert _image_distance(ldata, data[i]) == 0, "Error while encoding png image nbr {}".format(i)


def test_help_read_png():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        with open(filename, "rb") as f:
            ldata = read_png(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while reading png image nbr {}".format(i)


def test_help_write_png():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_png(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_png(f)
        assert _image_distance(ldata, data[i]) == 0, "Error while writing png image nbr {}".format(i)
    os.remove(filename)


def test_help_load_png():
    for i in range(len(data)):
        filename = FILENAMES_PNG.format(i)
        ldata = load_png(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while loading png image nbr {}".format(i)


def test_help_save_png():
    filename = FILENAMES_PNG.format("write")
    for i in range(len(data)):
        save_png(filename, data[i])
        ldata = load_png(filename)
        assert _image_distance(ldata, data[i]) == 0, "Error while saving png image nbr {}".format(i)
    os.remove(filename)


def test_help_examples_png():
    save_png("filename.png", data[0])
    import iotools.examples.examples_png  # noqa
    os.remove("filename.png")
