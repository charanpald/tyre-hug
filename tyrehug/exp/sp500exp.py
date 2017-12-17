import pandas
import os
import datetime
import numpy
import matplotlib.pyplot as plt
from os.path import expanduser

home = expanduser("~")
data_dir = os.path.join(home, "Data", "tyre-hug")
csv_filename = os.path.join(data_dir, "^GSPC.csv")


df = pandas.read_csv(csv_filename, index_col=0, parse_dates=True)

start = datetime.datetime(1950, 1, 3)
end = datetime.datetime(2017, 12, 15)

# Look at by week, month, year
# index = pandas.date_range(start, end, freq="W-MON")
index = pandas.date_range(start, end, freq="BM")
# index = pandas.date_range(start, end, freq="Y")

# Count days of the week with min price, and max price
maxes = []
mins = []

for i in range(len(index) - 1):
    df2 = df.loc[index[i]:index[i+1]][:-1]
    maxes.append(df2["Close"].values.argmax())
    mins.append(df2["Close"].values.argmin())


print(numpy.bincount(maxes))
print(numpy.bincount(mins))

#plt.figure(0)
plt.hist(maxes)

#plt.figure(1)
#plt.bar(numpy.arange(len(numpy.bincount(mins))), numpy.bincount(maxes) - numpy.bincount(mins))
plt.show()
