# iotools

`iotools` provides a unified syntax for reading and writing data to files.

Example:
```py
text = load_txt("filename.txt")
data = load_json("filename.json")
table = load_csv("filename.csv")
image = load_jpg("filename.jpg")
config = load_yaml("filename.yaml")
```


## Functions

### Reading

- `data = load_*(filename)`: filename is a path to an existing file
- `data = read_*(openfile)`: openfile is a file-like object
- `data = decode_*(rawbytes)`: rawbytes is the file's bytes


### Writing

- `save_*(filename, data)`: filename is a path to an existing file
- `write_*(openfile, data)`: openfile is a file-like object
- `rawbytes = encode_*(data)`: returns bytes


### Helping

- `help_*()`: prints simple examples and simple implementations on how to use this file format without iotools functions


## Installation

```bash
pip install iotools-wrappers
```


## Supported formats

- `avi`
- `bytes`: not really a format, but use this to load raw bytes
- `csv`
- `gif`
- `gz`: gzip
- `jpg`
- `json`
- `mp4`
- `npy`: numpy
- `pickle`
- `png`
- `tar`
- `txt`: text files
- `xml`
- `yaml`
- `zip`
- `zst`: zstandard


## How it works

`iotools` is nothing but wrappers on other python librairies, for example `load_json` will call `json.load` internally.
The backend used is specified in each function documentation.

`iotools` goal is to provide a memorable io syntax, on as many file formats as possible.

For more format-specific functions, or more complex io features, please use the library specific to that file format (`help_*()` may provide a good starting point)


## Examples

### Text

```py
from iotools import load_txt, read_txt

# Load text in one line:
text = load_txt("filename.txt")

# Or if you prefer the 'with open()' syntax:
with open("filename.txt", "r") as f:
     text = read_txt(f)  # same as f.read()
```

### Images

```py
# Example where image colors are inversed
from iotools import load_image, save_image

image = load_image("input.jpg")
save_image("output.jpg", image[:, :, ::-1])
```

### Archives

```py
# Load a tar.gz archive
from iotools import load_gz, decode_tar

data = decode_tar(load_gz("archive.tar.gz"))

# If your archive is made of images, you can even do:
from iotools import load_gz, decode_tar, decode_image

data = {filename: decode_image(img) for filename, img in decode_tar(load_gz("archive.tar.gz")).items()}
```

### GIFs

```py
import numpy as np
from iotools import save_gif

gif = [np.ones((180, 320), dtype=np.uint8) * i * 10 for i in range(26)]
save_gif("filename.gif", gif, fps=6)
```


## Additionnal features

### File systems

`iotools` is compatible with [fsspec](https://pypi.org/project/fsspec/ "fsspec"): `load_*` and `save_*` functions have an `fs` kwarg in order to use different file systems.

Example:
```py
# Load a tar.gz archive containing images
from iotools import load_image
from fsspec.implementations.tar import TarFileSystem

tarfs = TarFileSystem("archive.tar.gz")
data = {filename: load_image(filename, fs=tarfs) for filename in tarfs.glob("*")}
```

### Makedirs

`save_*(filename, data, fs=None, makedirs=False)` have a kwarg `makedirs` that allows to automatically create `filename`'s parent directory if it does not exist.

### Format specific options

Since `iotools` functions are mostly wrappers of other python libraires, most of formats specific options can be used through `iotools` functions.
For example you can do `save_json("filename.json", data, indent=4)`.

### Use your prefered library

Sometimes several python librairies can be used to load/save data to a format. To let you use your preferred library, `iotools` functions are declined in several forms with the keyword `using`. Here is an example with jpg images:
```py
from iotools.jpgio import load_jpg_using_imageio, load_jpg_using_pil, load_jpg_using_cv2

image = load_jpg_using_imageio("filename.jpg")  # uses imageio library
image = load_jpg_using_pil("filename.jpg")  # uses PIL (Pillow) library
image = load_jpg_using_cv2("filename.jpg")  # uses cv2 (opencv-python) library
```
Note that the output image can be different based on which library is used internally

### Generic formats

`iotools` supports 2 generic formats: images and videos. Thus there are functions like `load_image` or `load_video` to load any image or video format (formats not enumerated above were not tested).
`save_image` or `save_video` also exists, and a kwarg `format` is there for you to choose which image/video format to save in, otherwise it will be guessed from the filename.


## Other notes

### Import syntax

Several import syntaxs can be used:
```py
from iotools import load_csv
from iotools import csvio  # --> csvio.load_csv
from iotools.csvio import load_csv, load_csv_using_pandas
```

### OS Support

`iotools` was only tested under Linux. Under Windows, it seems that there are still issues with csvs, images and videos, but the other formats should be fine.

### Dependencies

Installing `iotools` does not install any format-specific backends, thus you may not be able to use a file format if the backend is not already installed. Please refer to `pyproject.toml` for backends supported versions.
