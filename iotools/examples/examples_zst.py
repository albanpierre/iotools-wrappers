
# Loading:
from iotools.zstio import load_zst
from iotools.jsonio import decode_json
raw_data = load_zst("filename.json.zst")
data = decode_json(raw_data)


# Saving:
from iotools.zstio import save_zst
from iotools.jsonio import encode_json
data = {"some": ["json", "data"]}
raw_data = encode_json(data)
save_zst("filename.json.zst", raw_data)
