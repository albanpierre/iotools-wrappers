
# Loading:
from iotools.jpgio import load_jpg_using_pil
data = load_jpg_using_pil("filename.jpg")
# data = np.asarray(data)


# Saving:
from iotools.jpgio import save_jpg_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_jpg_using_pil("filename.jpg", data)


# Saving #2:
from iotools.jpgio import save_jpg_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_jpg_using_pil("filename.jpg", data, quality=85)


# Saving #3:
from iotools.jpgio import save_jpg_using_pil
from PIL import Image
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
data = Image.fromarray(data)
save_jpg_using_pil("filename.jpg", {"image": data, "quality": 85})
