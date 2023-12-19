
# Loading:
from iotools.videoio import load_video_using_cv2
data = load_video_using_cv2("filename.mp4")
first_frame = data[0]


# Loading #2:
from iotools.videoio import load_video_using_cv2
data = load_video_using_cv2("filename.mp4", which="all")
first_frame = data["video"][0]
video_properties = data["properties"]


# Saving:
from iotools.videoio import save_video_using_cv2
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_video_using_cv2("filename.mp4", frames)


# Saving #2:
from iotools.videoio import save_video_using_cv2
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
save_video_using_cv2("filename.mp4", frames, fps=6, format="mp4")


# Saving #3:
from iotools.videoio import save_video_using_cv2
import numpy as np
frames = [i * 10 * np.ones((160, 240, 3), dtype=np.uint8) for i in range(25)]
data = {"video": frames, "properties": {"fps": 6, "format": "mp4"}}
save_video_using_cv2("filename.mp4", data)
