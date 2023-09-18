
# Loading:
from iotools.txtio import load_txt
data = load_txt("filename.txt")


# Saving:
from iotools.txtio import save_txt
data = "some_text"
save_txt("filename.txt", data)
