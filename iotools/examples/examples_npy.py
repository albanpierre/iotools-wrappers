
# Loading:
from iotools.npyio import load_npy
data = load_npy("filename.npy", allow_pickle=True)


# Saving:
from iotools.npyio import save_npy
import numpy as np
data = np.asarray([1, 2, 3])
save_npy("filename.npy", data)
