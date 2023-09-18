
# Loading:
from iotools.gzio import load_gz
from iotools.tario import decode_tar
raw_data = load_gz("filename.tar.gz")
data = decode_tar(raw_data)


# Saving:
from iotools.gzio import save_gz
from iotools.tario import encode_tar
data = {"subfilename.txt": "some_text"}
raw_data = encode_tar(data)
save_gz("filename.tar.gz", raw_data)
