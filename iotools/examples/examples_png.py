
# Loading:
from iotools.pngio import load_png
data = load_png("filename.png")


# Saving:
from iotools.pngio import save_png
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_png("filename.png", data)
