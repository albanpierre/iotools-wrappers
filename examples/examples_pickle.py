
# Loading:
from iotools.pickleio import load_pickle
data = load_pickle("filename.pickle")


# Saving:
from iotools.pickleio import save_pickle
data = {"some": ("pickle", b"data")}
save_pickle("filename.pickle", data)
