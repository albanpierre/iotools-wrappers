
# Loading:
from iotools.aviio import load_avi
data = load_avi("filename.avi")
first_frame = data[0]


# Loading #2:
from iotools.aviio import load_avi
data = load_avi("filename.avi", which="all")
first_frame = data["video"][0]
video_properties = data["properties"]


# Saving:
from iotools.aviio import save_avi
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_avi("filename.avi", frames)


# Saving #2:
from iotools.aviio import save_avi
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_avi("filename.avi", frames, fps=6)


# Saving #3:
from iotools.aviio import save_avi
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
data = {"video": frames, "properties": {"fps": 6}}
save_avi("filename.avi", data)
