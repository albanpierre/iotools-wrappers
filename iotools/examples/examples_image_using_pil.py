
# Loading:
from iotools.imageio import load_image_using_pil
data = load_image_using_pil("filename.png")
# data = np.asarray(data)


# Saving:
from iotools.imageio import save_image_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_image_using_pil("filename.png", data)


# Saving #2:
from iotools.imageio import save_image_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_image_using_pil("filename.jpg", data, format="jpg", quality=85)


# Saving #3:
from iotools.imageio import save_image_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_image_using_pil("filename.jpg", {"image": data, "format": "jpg", "quality": 85})
