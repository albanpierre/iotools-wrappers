
# Loading:
from iotools.pngio import load_png_using_pil
data = load_png_using_pil("filename.png")
# data = np.asarray(data)


# Saving:
from iotools.pngio import save_png_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_png_using_pil("filename.png", data)
