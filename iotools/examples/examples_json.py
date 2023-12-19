
# Loading:
from iotools.jsonio import load_json
data = load_json("filename.json")


# Saving:
from iotools.jsonio import save_json
data = {"some": ["json", "data"]}
save_json("filename.json", data)


# Saving #2:
from iotools.jsonio import save_json
data = {"some": ["json", "data"]}
save_json("filename.json", data, indent=4)
