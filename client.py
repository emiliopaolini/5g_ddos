


import tensorflow as tf
import flwr as fl
import hickle as hkl
import numpy as np
import sklearn.metrics
import argparse
import os


from pathlib import Path
from model import *
from tensorflow.keras.optimizers import Adam

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        prediction = (self.model.predict(self.x_test) > 0.5).astype("int32")
        
        
        f1 = np.round(sklearn.metrics.f1_score(self.y_test, prediction), 5)
        acc = np.round(sklearn.metrics.accuracy_score(self.y_test, prediction), 5)
        precision = np.round(sklearn.metrics.precision_score(self.y_test, prediction), 5)
        recall = np.round(sklearn.metrics.recall_score(self.y_test, prediction), 5)

        loss, _ = self.model.evaluate(self.x_test, self.y_test, 32)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load and compile Keras model

    model = Conv2D_1()

    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer= opt, loss='binary_crossentropy',metrics='accuracy')
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition

    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    data = hkl.load('./data.hkl')

    X_train = data['xtrain']
    y_train = data['ytrain']

    X_test = data['xtest']
    y_test = data['ytest']

    return (
        X_train[idx * 4000 : (idx + 1) * 4000],
        y_train[idx * 4000 : (idx + 1) * 4000],
    ), (
        X_test[idx * 800 : (idx + 1) * 800],
        y_test[idx * 800 : (idx + 1) * 800],
    )


if __name__ == "__main__":
    main()









'''
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam


import sklearn

checkpoint_filepath = '/tmp/checkpoint'

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True,
    verbose=True)


net = autoencoderConv2D_1()

print(X_train.shape)

print(net.summary())
opt = Adam(learning_rate=0.01)
net.compile(optimizer= opt, loss='binary_crossentropy',metrics='accuracy')
history = net.fit(X_train, y_train, batch_size = 64, shuffle=True,epochs = 10, verbose = True,callbacks=[model_checkpoint_callback])

net.load_weights(checkpoint_filepath)

prediction = (net.predict(X_test) > 0.5).astype("int32")


f1_score = np.round(sklearn.metrics.f1_score(y_test, prediction), 5)
acc = np.round(sklearn.metrics.accuracy_score(y_test, prediction), 5)
precision = np.round(sklearn.metrics.precision_score(y_test, prediction), 5)
recall = np.round(sklearn.metrics.recall_score(y_test, prediction), 5)

print('Acc = %.5f, f1_score = %.5f, precision = %.5f, recall = %.5f' % (acc,f1_score,precision,recall))
'''