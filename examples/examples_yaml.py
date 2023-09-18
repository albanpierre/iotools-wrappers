
# Loading:
from iotools.yamlio import load_yaml
data = load_yaml("filename.yaml")


# Saving:
from iotools.yamlio import save_yaml
data = {"some": ["yaml", "data"]}
save_yaml("filename.yaml", data)
