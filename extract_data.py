import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


base = os.path.abspath(os.path.dirname(__file__))

def load_data():
    global base
    def get_label_from_path(path):
        fname = os.path.basename(path)
        label = fname.split(".")[0]
        return [1, 0] if label == "cat" else [0, 1]

    imsize = (64, 64)
    traindir = os.path.join(base, "data", "dataset")
    files = map(lambda x: os.path.join(traindir, x), os.listdir(traindir))
    X, y = [], []

    for _f in files:
        # img = Image.open(_f).resize(imsize).convert("L")
        img = Image.open(_f).resize(imsize)

        X.append(np.array(img))
        y.append(get_label_from_path(_f))

    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Extracting...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    with h5py.File(os.path.join(base, "data", "dataset.hdf5"), "w") as f:
        train_grp = f.create_group("train")
        test_grp = f.create_group("test")    
        train_grp["X"], train_grp["y"] = X_train, y_train
        test_grp["X"], test_grp["y"] = X_test, y_test
    print("Done")