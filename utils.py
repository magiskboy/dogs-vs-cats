import os
import pandas as pd
import itertools
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


basedir = os.path.abspath(os.path.dirname(__file__))
traindir = os.path.join(basedir, "data", "train")
testdir = os.path.join(basedir, "data", "test")

def files_iter():
    global traindir
    global testdir
    file_filter = lambda x: os.path.isfile(x)
    test_files = [os.path.join(testdir, _x) for _x in os.listdir(testdir)]
    train_files = [os.path.join(traindir, _x) for _x in os.listdir(traindir)]
    return filter(file_filter, train_files), filter(file_filter, test_files)

def extract_sample(flatten=True, gray=True, basesize=(28, 28)):
    while True:
        filepath = yield
        filename = os.path.basename(filepath)
        label = filename.split(".")[0]
        image = Image.open(filepath).convert("L") if gray else Image.open(filepath)
        image = image.resize(basesize)
        arr = np.array(image)
        yield (label, arr.reshape(-1)) if flatten else (label, arr)

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def create_model(model, params={}):
    steps = [
        ("standard", preprocessing.StandardScaler()),
        ("normalize", preprocessing.MinMaxScaler()),
        ("model", model())
    ]
    pipe = Pipeline(steps)
    pipe.set_params(**params)
    return pipe

def encode_label(labels):
    map_fn = lambda label: 1 if label == "cat" else 0
    labels_encoded = list(map(map_fn, labels))
    return labels_encoded

def decode_label(labels_encoded):
    map_fn = lambda label_encoded: "cat" if labels_encoded == 1 else "dog"
    labels = list(map(map_fn, labels_encoded))
    return labels

def test(pipeline, test_set, batch_size):
    s = 0
    n = 0
    extract_data_ops = extract_sample()
    next(extract_data_ops)
    bucket = {"X": [], "y": []}

    for _iter, test_sample in enumerate(test_set):
        n += 1
        label, pixels = extract_data_ops.send(test_sample)
        bucket["X"].append(pixels)
        bucket["y"].append(label)
        next(extract_data_ops)
        if _iter % batch_size == 0:
            X = bucket["X"]
            y = encode_label(bucket["y"])
            pred = pipeline.predict(X)
            for i in range(len(pred)):
                s += int(pred[i] == y[i])
            bucket["X"].clear()
            bucket["y"].clear()

    return (s / n)

def train(pipeline, train_set, test_set, batch_size):
    # Create operator for extract data from filepath of image
    extract_data_ops = extract_sample()
    next(extract_data_ops)
    bucket = {"X": [], "y": []}
    accuracy = []

    for _iter, train_sample in enumerate(train_set):
        label, pixels = extract_data_ops.send(train_sample)
        next(extract_data_ops)
        bucket["X"].append(pixels)
        bucket["y"].append(label)

        if len(bucket["X"]) == batch_size:
            X = bucket["X"]
            y = encode_label(bucket["y"])
            pipeline.fit(X, y)
            # test_set, test_set_clone = itertools.tee(test_set)
            # acc = test(pipeline, test_set_clone, batch_size)
            # accuracy.append(acc)
            bucket["X"].clear()
            bucket["y"].clear()
            print("Iter %d" % _iter)

    return pipeline, accuracy

if __name__ == "__main__":
    train_iter, test_iter = files_iter()
    extract_node = extract_sample(basesize=(64, 64))
    next(extract_node)
    train, test = [], []

    for i, _f in enumerate(train_iter):
        if i % 1000 == 0:
            print(i)
        row, label = extract_node.send(_f)
        next(extract_node)
        row = np.append(row, label)
        train.append(row)
    df = pd.DataFrame(train, index=None, columns=None)
    df.to_csv("train.csv")

    del train, df

    for i, _f in enumerate(test_iter):
        if i % 1000 == 0:
            print(i)
        row, label = extract_node.send(_f)
        next(extract_node)
        row = np.append(row, label)
        test.append(row)
    df = pd.DataFrame(test, index=None, columns=None)
    df.to_csv("test.csv")
