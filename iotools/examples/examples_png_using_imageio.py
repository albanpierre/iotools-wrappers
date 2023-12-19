
# Loading:
from iotools.pngio import load_png_using_imageio
data = load_png_using_imageio("filename.png")


# Saving:
from iotools.pngio import save_png_using_imageio
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_png_using_imageio("filename.png", data)
