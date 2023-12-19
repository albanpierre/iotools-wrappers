
# Loading:
from iotools.csvio import load_csv_using_csv
data = load_csv_using_csv("filename.csv")


# Saving:
from iotools.csvio import save_csv_using_csv
data = [[2, "some"], [3, "csv"]]
save_csv_using_csv("filename.csv", data)
