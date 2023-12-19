
# Loading:
from iotools.tario import load_tar
data = load_tar("filename.tar")  # data is a dict of the form {subfilename: bytes}


# Saving:
from iotools.tario import save_tar
data = {"subfilename.txt": "some_text"}
save_tar("filename.tar", data)
