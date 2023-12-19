
# Loading:
from iotools.gifio import load_gif
data = load_gif("filename.gif")
first_frame = data[0]


# Loading #2:
from iotools.gifio import load_gif
data = load_gif("filename.gif", which="all")
first_frame = data["video"][0]
video_properties = data["properties"]


# Saving:
from iotools.gifio import save_gif
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_gif("filename.gif", frames)


# Saving #2:
from iotools.gifio import save_gif
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_gif("filename.gif", frames, fps=6)


# Saving #3:
from iotools.gifio import save_gif
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
data = {"video": frames, "properties": {"fps": 6}}
save_gif("filename.gif", data)
