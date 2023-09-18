
# Loading:
from iotools.jpgio import load_jpg_using_imageio
data = load_jpg_using_imageio("filename.jpg")


# Saving:
from iotools.jpgio import save_jpg_using_imageio
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_jpg_using_imageio("filename.jpg", data)


# Saving #2:
from iotools.jpgio import save_jpg_using_imageio
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_jpg_using_imageio("filename.jpg", data, quality=85)


# Saving #3:
from iotools.jpgio import save_jpg_using_imageio
import numpy as np
data = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[:, np.newaxis, np.newaxis], 100, axis=1), 3, axis=2)
save_jpg_using_imageio("filename.jpg", {"image": data, "quality": 85})
