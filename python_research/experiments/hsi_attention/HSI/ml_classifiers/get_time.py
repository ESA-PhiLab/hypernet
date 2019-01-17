import os
import numpy as np

PATH = r"C:\Users\mmarcinkiewicz\Desktop\HSI_artifacts_ml"

timestamps = list()
files = list()
for file in [x for x in os.listdir(PATH) if "pc" not in x]:
    timestamps.append(os.path.getmtime(os.path.join(PATH, file)))
    files.append(file.split("_acc")[0])
timestamps = np.array(timestamps)
timestamps -= timestamps.min()
timestamps = list(timestamps)


a = sorted(zip(files, timestamps), key= lambda pair: pair[1])
last_timestamp = 0
for path, timestamp in a:
    print(path, timestamp - last_timestamp)
    last_timestamp = timestamp
print()