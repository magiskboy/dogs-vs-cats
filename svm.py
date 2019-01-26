import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from utils import extract_sample
from utils import files_iter
from utils import create_model
from utils import save_model, load_model
from utils import encode_label
from utils import train, test


epochs = 1

# get generator of list filename
train_set, test_set = files_iter()

# create model
# pipeline = create_model(svm.SVC)

# for epoch in range(epochs):
#     print("Epoch %d" % epoch)
#     train_set, train_set_clone = itertools.tee(train_set)
#     pipeline, accuracy = train(pipeline, train_set_clone, test_set, 17500)

# save_model(pipeline, "svm.joblib")

# plt.plot(accuracy)
# plt.title("Accuracy for SVM")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.show()
# print("Accuracy %.3f" % accuracy[-1])

pipeline = load_model("svm.joblib")
acc = test(pipeline, test_set, 2500)
print("Accuracy %.3f" % acc)