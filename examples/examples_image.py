
# Loading:
from iotools.imageio import load_image
data = load_image("filename.png")


# Saving:
from iotools.imageio import save_image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_image("filename.png", data)


# Saving #2:
from iotools.imageio import save_image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_image("filename.jpg", data, format="jpg", quality=85)


# Saving #3:
from iotools.imageio import save_image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_image("filename.jpg", {"image": data, "format": "jpg", "quality": 85})
