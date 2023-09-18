
import os
import numpy as np
import PIL.Image as PilImage

from iotools.bytesio import load_bytes
from simplified_iotools.imageio_using_imageio import (
    decode_image_using_imageio, encode_image_using_imageio,
    read_image_using_imageio, write_image_using_imageio,
    load_image_using_imageio, save_image_using_imageio
)
from simplified_iotools.imageio_using_skimage import (
    decode_image_using_skimage, encode_image_using_skimage,
    read_image_using_skimage, write_image_using_skimage,
    load_image_using_skimage, save_image_using_skimage
)
from simplified_iotools.imageio_using_pil import (
    decode_image_using_pil, encode_image_using_pil,
    read_image_using_pil, write_image_using_pil,
    load_image_using_pil, save_image_using_pil
)
from simplified_iotools.imageio_using_cv2 import (
    decode_image_using_cv2, encode_image_using_cv2,
    read_image_using_cv2, write_image_using_cv2,
    load_image_using_cv2, save_image_using_cv2
)
from simplified_iotools.imageio import (
    decode_image, encode_image, read_image, write_image, load_image, save_image
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_JPG = os.path.join(TEST_DATA_PATH, "image", "example_jpg_{}.jpg")
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

data_jpg = {}
data_png = {}
data_jpg[0] = image[0][:, :, :3].copy()
data_png[0] = image[0].copy()


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


read_list_of_args = list(zip(
    [FILENAMES_JPG, FILENAMES_PNG],
    [data_jpg, data_png],
    ["jpg", "png"],
))

write_list_of_args = list(zip(
    [FILENAMES_JPG, FILENAMES_PNG],
    [data_jpg, data_png],
    ["jpg", "png"],
    [{}, {}],
))


# +---------------+
# | Using imageio |
# +---------------+


def test_help_decode_image_using_imageio():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            txt = load_bytes(filename)
            ldata = decode_image_using_imageio(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while decoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while decoding image file nbr {}{}".format(i, ext)


def test_help_encode_image_using_imageio():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        for i in range(len(data)):
            txt = encode_image_using_imageio(data[i], format=ext, **kwargs)
            ldata = decode_image_using_imageio(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while encoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while encoding image file nbr {}{}".format(i, ext)


def test_help_read_image_using_imageio():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            with open(filename, "rb") as f:
                ldata = read_image_using_imageio(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while reading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while reading image file nbr {}{}".format(i, ext)


def test_help_write_image_using_imageio():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_image_using_imageio(f, data[i], format=ext, **kwargs)
            with open(filename, "rb") as f:
                ldata = read_image_using_imageio(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while writing image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while writing image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_load_image_using_imageio():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            ldata = load_image_using_imageio(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while loading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while loading image file nbr {}{}".format(i, ext)


def test_help_save_image_using_imageio():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            save_image_using_imageio(filename, data[i], **kwargs)
            ldata = load_image_using_imageio(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while saving image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while saving image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_examples_image_using_imageio():
    save_image_using_imageio("filename.png", data_png[0])
    save_image_using_imageio("filename.jpg", data_jpg[0])
    import examples.examples_image_using_imageio  # noqa
    os.remove("filename.png")
    os.remove("filename.jpg")


# +---------------+
# | Using skimage |
# +---------------+


def test_help_decode_image_using_skimage():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            txt = load_bytes(filename)
            ldata = decode_image_using_skimage(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while decoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while decoding image file nbr {}{}".format(i, ext)


def test_help_encode_image_using_skimage():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        for i in range(len(data)):
            txt = encode_image_using_skimage(data[i], format=ext, **kwargs)
            ldata = decode_image_using_skimage(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while encoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while encoding image file nbr {}{}".format(i, ext)


def test_help_read_image_using_skimage():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            with open(filename, "rb") as f:
                ldata = read_image_using_skimage(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while reading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while reading image file nbr {}{}".format(i, ext)


def test_help_write_image_using_skimage():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_image_using_skimage(f, data[i], format=ext, **kwargs)
            with open(filename, "rb") as f:
                ldata = read_image_using_skimage(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while writing image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while writing image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_load_image_using_skimage():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            ldata = load_image_using_skimage(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while loading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while loading image file nbr {}{}".format(i, ext)


def test_help_save_image_using_skimage():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            save_image_using_skimage(filename, data[i], **kwargs)
            ldata = load_image_using_skimage(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while saving image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while saving image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_examples_image_using_skimage():
    save_image_using_skimage("filename.png", data_png[0])
    save_image_using_skimage("filename.jpg", data_jpg[0])
    import examples.examples_image_using_skimage  # noqa
    os.remove("filename.png")
    os.remove("filename.jpg")


# +-----------+
# | Using PIL |
# +-----------+


def test_help_decode_image_using_pil():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            txt = load_bytes(filename)
            ldata = decode_image_using_pil(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while decoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while decoding image file nbr {}{}".format(i, ext)


def test_help_encode_image_using_pil():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        for i in range(len(data)):
            if (ext == "jpg") and (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            txt = encode_image_using_pil(PilImage.fromarray(data[i]), format=ext, **kwargs)
            ldata = decode_image_using_pil(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while encoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while encoding image file nbr {}{}".format(i, ext)


def test_help_read_image_using_pil():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            with open(filename, "rb") as f:
                ldata = read_image_using_pil(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while reading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while reading image file nbr {}{}".format(i, ext)


def test_help_write_image_using_pil():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            if (ext == "jpg") and (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            with open(filename, "wb") as f:
                write_image_using_pil(f, PilImage.fromarray(data[i]), format=ext, **kwargs)
            with open(filename, "rb") as f:
                ldata = read_image_using_pil(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while writing image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while writing image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_load_image_using_pil():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            ldata = load_image_using_pil(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while loading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while loading image file nbr {}{}".format(i, ext)


def test_help_save_image_using_pil():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            if (ext == "jpg") and (data[i].ndim == 3) and (data[i].shape[2] == 4):
                continue  # do not test RGBA jpg images since PIL will raise an error in that case
            save_image_using_pil(filename, PilImage.fromarray(data[i]), **kwargs)
            ldata = load_image_using_pil(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while saving image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while saving image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_examples_image_using_pil():
    save_image_using_pil("filename.png", PilImage.fromarray(data_png[0]))
    save_image_using_pil("filename.jpg", PilImage.fromarray(data_jpg[0]))
    import examples.examples_image_using_pil  # noqa
    os.remove("filename.png")
    os.remove("filename.jpg")


# +-----------+
# | Using cv2 |
# +-----------+


def test_help_decode_image_using_cv2():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            txt = load_bytes(filename)
            ldata = decode_image_using_cv2(txt)
            ldata = _format_image(ldata)
            ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while decoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while decoding image file nbr {}{}".format(i, ext)


def test_help_encode_image_using_cv2():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        for i in range(len(data)):
            txt = encode_image_using_cv2(data[i], format=ext, **kwargs)
            ldata = decode_image_using_cv2(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while encoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while encoding image file nbr {}{}".format(i, ext)


def test_help_read_image_using_cv2():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            with open(filename, "rb") as f:
                ldata = read_image_using_cv2(f)
            ldata = _format_image(ldata)
            ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while reading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while reading image file nbr {}{}".format(i, ext)


def test_help_write_image_using_cv2():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_image_using_cv2(f, data[i], format=ext, **kwargs)
            with open(filename, "rb") as f:
                ldata = read_image_using_cv2(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while writing image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while writing image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_load_image_using_cv2():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            ldata = load_image_using_cv2(filename)
            ldata = _format_image(ldata)
            ldata[:, :, :3] = ldata[:, :, 2::-1]  # cv2 inverses colors (BGR instead of RGB)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while loading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while loading image file nbr {}{}".format(i, ext)


def test_help_save_image_using_cv2():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            save_image_using_cv2(filename, data[i], **kwargs)
            ldata = load_image_using_cv2(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while saving image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while saving image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_examples_image_using_cv2():
    save_image_using_cv2("filename.png", data_png[0])
    save_image_using_cv2("filename.jpg", data_jpg[0])
    import examples.examples_image_using_cv2  # noqa
    os.remove("filename.png")
    os.remove("filename.jpg")


# +------------------+
# | Default behavior |
# +------------------+


def test_help_decode_image():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            txt = load_bytes(filename)
            ldata = decode_image(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while decoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while decoding image file nbr {}{}".format(i, ext)


def test_help_encode_image():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        for i in range(len(data)):
            txt = encode_image(data[i], format=ext, **kwargs)
            ldata = decode_image(txt)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while encoding image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while encoding image file nbr {}{}".format(i, ext)


def test_help_read_image():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            with open(filename, "rb") as f:
                ldata = read_image(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while reading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while reading image file nbr {}{}".format(i, ext)


def test_help_write_image():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_image(f, data[i], format=ext, **kwargs)
            with open(filename, "rb") as f:
                ldata = read_image(f)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while writing image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while writing image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_load_image():
    for filenames_image, data, ext in read_list_of_args:
        for i in range(len(data)):
            filename = filenames_image.format(i)
            ldata = load_image(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while loading image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while loading image file nbr {}{}".format(i, ext)


def test_help_save_image():
    for filenames_image, data, ext, kwargs in write_list_of_args:
        filename = filenames_image.format("write")
        for i in range(len(data)):
            save_image(filename, data[i], **kwargs)
            ldata = load_image(filename)
            if (ext == "jpg"):
                assert _image_distance(ldata, data[i]) < 32, "Error while saving image file nbr {}{}".format(i, ext)
            else:
                assert _image_distance(ldata, data[i]) == 0, "Error while saving image file nbr {}{}".format(i, ext)
        os.remove(filename)


def test_help_examples_image():
    save_image("filename.png", data_png[0])
    save_image("filename.jpg", data_jpg[0])
    import examples.examples_image  # noqa
    os.remove("filename.png")
    os.remove("filename.jpg")
