from pathlib import Path
import os

input_images_path = "\Images"
filesNames = os.listdir(input_images_path)

vector = range(len(filesNames))
print(vector)
root = "Preliminary Results\\"

for i in vector:
    file = filesNames[i]
    path = root + file
    print(path)
    directory = Path(path)
    directory.mkdir(parents=True)
