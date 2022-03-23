from tensorflow import keras

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
from scikeras.wrappers import KerasRegressor

from sklearn import metrics
from sklearn.model_selection import cross_val_predict

import numpy as np

import mlflow
import hyperopt


def model_creator(
    input_shape,
    num_units=2,
    num_layers=0,
    kernel_regularizer="l2",
    kernel_weight=10**-4,
    activation="relu",
    loss="categorical_crossentropy",
    learning_rate=0.01
):
    kernel_regularizer = l2 if kernel_regularizer == "l2" else l1


    model = Sequential()
    model.add(Input(shape=(input_shape[1])))

    for _ in range(num_layers):
        model.add(Dense(units=num_units, activation=activation, kernel_regularizer=kernel_regularizer(kernel_weight)))
    
    # output layer
    model.add(Dense(2,  activation="softmax", kernel_regularizer=kernel_regularizer(kernel_weight)))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=loss, metrics=['mse'])

    return model

def regression_metrics(actual, pred):
    return {
        "MAE": metrics.mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(metrics.mean_absolute_error(actual, pred)),
        "MSE": metrics.mean_squared_error(actual, pred),
    }

def fit_and_log_cv(x_train, y_train, x_test, y_test, params, nested=False):
    with mlflow.start_run(nested=nested, experiment_id=1) as run:
        print(params)
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        model_cv = KerasRegressor(
            model=model_creator,
            input_shape=x_train.shape, 
            batch_size=32,
            **params
        )
        
        y_pred_cv = cross_val_predict(model_cv, x_train, y_train)
        # print(y_pred_cv.shape)
        # print(y_pred_cv)
        metrics_cv = {f"val_{metric}": value for metric, value in regression_metrics(y_train, y_pred_cv).items()}

        mlflow.tensorflow.autolog()
        model = KerasRegressor(
            model=model_creator,
            input_shape=x_train.shape, 
            **params
        )
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        metrics_test = {f"test_{metric}": value for metric, value in regression_metrics(y_test, y_pred_test).items()}

        metrics = {**metrics_test, **metrics_cv}
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        return metrics

def mse(x, y):
    return ((x-y)**2).mean()

def log_best(run: mlflow.entities.Run,
         metric: str) -> None:
        """Log the best parameters from optimization to the parent experiment.

        Args:
            run: current run to log metrics
            metric: name of metric to select best and log
        """

        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            [run.info.experiment_id],
            "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))

        best_run = min(runs, key=lambda run: run.data.metrics[metric])

        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])

def train_model_v2( 
        x_train,
        y_train,
        x_test,
        y_test,
        metric:str
    ):
        def train_func(params):
            metrics = fit_and_log_cv(x_train, y_train, x_test, y_test, params, nested=True)
            print(metrics)
            print({'status': hyperopt.STATUS_OK, 'loss': metrics[metric]})
            return {'status': hyperopt.STATUS_OK, 'loss': float(metrics[metric])}
        
        return train_func