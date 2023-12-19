
# Loading:
from iotools.bytesio import load_bytes
data = load_bytes("filename.bytes")


# Saving:
from iotools.bytesio import save_bytes
data = b"some_binary_text"
save_bytes("filename.bytes", data)
