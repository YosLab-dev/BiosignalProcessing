import pandas as pd
from pathlib import Path

dFrame = pd.read_excel('Database.xlsx')
pathD = dFrame['Path']
vector = range(len(pathD))
print(vector)
root = "Preliminary Results\ "

for i in vector:
    file = pathD[i]
    path = root + file
    print(path)
    directory = Path(path)
    directory.mkdir(parents=True)
