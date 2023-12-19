
# Loading:
from iotools.mp4io import load_mp4_using_imageio
data = load_mp4_using_imageio("filename.mp4")
first_frame = data[0]


# Loading #2:
from iotools.mp4io import load_mp4_using_imageio
data = load_mp4_using_imageio("filename.mp4", which="all")
first_frame = data["video"][0]
video_properties = data["properties"]


# Saving:
from iotools.mp4io import save_mp4_using_imageio
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_mp4_using_imageio("filename.mp4", frames)


# Saving #2:
from iotools.mp4io import save_mp4_using_imageio
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_mp4_using_imageio("filename.mp4", frames, fps=6)


# Saving #3:
from iotools.mp4io import save_mp4_using_imageio
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
data = {"video": frames, "properties": {"fps": 6}}
save_mp4_using_imageio("filename.mp4", data)
