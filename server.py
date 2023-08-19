from typing import Dict, Optional, Tuple
from pathlib import Path
from model import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score 
from typing import List, Tuple
from flwr.common import Metrics

import flwr as fl
import tensorflow as tf
import hickle as hkl
import numpy as np
import sklearn

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Conv2D_1()

    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer= opt, loss='binary_crossentropy',metrics='accuracy')

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=1,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy_aggregated": sum(accuracies) / sum(examples), "f1_aggregated": sum(f1s) / sum(examples),
            "precision_aggregated": sum(precisions) / sum(examples),"recall_aggregated": sum(recalls) / sum(examples)}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    

    # Use the last 5k training examples as a validation set

    data = hkl.load('./data.hkl')

    X_val = data['xtest']
    y_val = data['ytest']

    x_val, y_val = X_val[2400:3200], y_val[2400:3200]

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)

        prediction = (model.predict(x_val) > 0.5).astype("int32")
        
        f1_score = np.round(sklearn.metrics.f1_score(y_val, prediction), 5)

        acc = np.round(sklearn.metrics.accuracy_score(y_val, prediction), 5)
        precision = np.round(sklearn.metrics.precision_score(y_val, prediction), 5)
        recall = np.round(sklearn.metrics.recall_score(y_val, prediction), 5)
        return loss, {"accuracy": acc, "f1": f1_score, "precision": precision, "recall": recall}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 5,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()