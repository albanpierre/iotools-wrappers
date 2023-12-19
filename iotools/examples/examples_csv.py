
# Loading:
from iotools.csvio import load_csv
data = load_csv("filename.csv")


# Saving:
from iotools.csvio import save_csv
data = [[2, "some"], [3, "csv"]]
save_csv("filename.csv", data)
