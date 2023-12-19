
# Loading:
from iotools.csvio import load_csv_using_pandas
data = load_csv_using_pandas("filename.csv")  # returns a pandas.DataFrame()


# Loading #2:
from iotools.csvio import load_csv_using_pandas
data = load_csv_using_pandas("filename.csv", keep_default_na=False)  # returns a pandas.DataFrame()


# Saving:
from iotools.csvio import save_csv_using_pandas
import pandas as pd
data = pd.DataFrame([[2, "some"], [3, "csv"]])
save_csv_using_pandas("filename.csv", data)


# Saving #2:
from iotools.csvio import save_csv_using_pandas
import pandas as pd
data = pd.DataFrame([[2, "some"], [3, "csv"]])
save_csv_using_pandas("filename.csv", data, index=False, header=False)
