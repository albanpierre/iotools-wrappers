
# Loading:
from iotools.zipio import load_zip
data = load_zip("filename.zip")  # data is a dict of the form {subfilename: bytes}


# Saving:
from iotools.zipio import save_zip
data = {"subfilename.txt": "some_text"}
save_zip("filename.zip", data)
