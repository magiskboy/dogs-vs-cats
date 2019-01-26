import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


nClasses = 2
parser = argparse.ArgumentParser(description="Dog vs Cat in kaggle")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training")
parser.add_argument("--epochs", type=int, default=20, help="Number of iterators")
parser.add_argument("--bsize", type=int, default=512, help="Size of mini batch")
arg = parser.parse_args()

def load_data():
    datafile = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "dataset.hdf5")
    f = h5py.File(datafile, "r")
    x_train, y_train = f["train/X"][()], f["train/y"][()]
    x_test, y_test = f["test/X"][()], f["test/y"][()]
    return x_train, y_train, x_test, y_test

def create_placeholder(width, height, channel):
    global nClasses
    X = tf.placeholder(shape=[None, height, width, channel], dtype=tf.float64)
    y = tf.placeholder(shape=[None, nClasses], dtype=tf.float32)
    return X, y

def initialize_parameters(channel):
    W1 = tf.get_variable("W1", [5, 5, channel, 32], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, 32, 64], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [5, 5, 64, 128], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=2))
    W4 = tf.get_variable("W4", [5, 5, 128, 64], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=3))
    W5 = tf.get_variable("W5", [5, 5, 64, 32], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=4))
    return W1, W2, W3, W4, W5

def forward_net(X, W):
    global nClasses
    W1, W2, W3, W4, W5 = W

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding="SAME")
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

    Z5 = tf.nn.conv2d(P4, W5, strides=[1, 1, 1, 1], padding="SAME")
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

    F1 = tf.contrib.layers.flatten(P5)
    F1 = tf.contrib.layers.fully_connected(F1, 1024, activation_fn=tf.nn.relu)
    F1 = tf.layers.dropout(F1, rate=0.5)

    predictor = tf.contrib.layers.fully_connected(F1, nClasses, activation_fn=tf.nn.softmax)

    return predictor

def compute_cost(predictor, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictor, labels=y))
    return cost

def random_mini_batches(X, y, n, seed):
    np.random.seed(seed)
    X, y = shuffle(X, y, random_state=seed)
    Xs = np.array_split(X, n)
    ys = np.array_split(y, n)
    minibatches = []
    for i in range(len(Xs)):
        minibatches.append((Xs[i], ys[i]))
    return minibatches

def model(X_train, y_train, X_test, y_test, lr=0.09, num_epochs=10, minibatch_size=512, print_cost=True):
    m, imHeight, imWidth, imChannel = X_train.shape
    costs = []
    
    X, y = create_placeholder(imWidth, imHeight, imChannel)
    W = initialize_parameters(imChannel)
    predictor = forward_net(X, W)
    cost = compute_cost(predictor, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            if num_minibatches == 0:
                num_minibatches = 1
            minibatches = random_mini_batches(X_train, y_train, num_minibatches, seed=epoch)

            for minibatch in minibatches:
                minibatch_X, minibatch_y = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, y: minibatch_y})
                minibatch_cost += temp_cost / num_minibatches
            
            # saver.save(sess, 'model_iter', global_step=epoch)
            # Print the cost every epoch
            costs.append(minibatch_cost)
            if print_cost:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

        saver.save(sess, os.path.join(os.path.dirname(__file__), "models", "cnn", "model.tf"))

        # Calculate the correct predictions
        predict_op = tf.argmax(predictor, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        
        # Calculate accuracy on the test set
        minibatches = random_mini_batches(X_train, y_train, num_minibatches, 7)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy, test_accuracy = [], []
        for minibatch in minibatches:
            X_p, y_p = minibatch
            train_accuracy.append(accuracy.eval({X: X_p, y: y_p}))

        minibatches = random_mini_batches(X_test, y_test, num_minibatches, 7)
        for minibatch in minibatches:
            X_p, y_p = minibatch
            test_accuracy.append(accuracy.eval({X: X_p, y: y_p}))
        
        print("Train accuracy", sum(train_accuracy) / len(train_accuracy))
        print("Test accuracy", sum(test_accuracy) / len(test_accuracy))

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(lr))
        plt.show()
            
    return train_accuracy, test_accuracy, W

if __name__ == "__main__":
    print("Training dog_vs_cat by CNN")
    print("Loading data to memory...")
    X_train, y_train, X_test, y_test = load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # X_train = X_train[:100] / 255.0
    # y_train = y_train[:100]
    # X_test = X_test[:10] / 255.0
    # y_test = y_test[:10]

    print("Load data complete")
    print("Train dataset: {}".format(X_train.shape))
    print("Test dataset: {}".format(X_test.shape))
    print("Starting train...")
    _, _, W = model(
        X_train, y_train, X_test, y_test,
        lr=arg.lr,
        num_epochs=arg.epochs,
        minibatch_size=arg.bsize
    )