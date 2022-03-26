import faulthandler
import os
import sys
import numpy as np
from pyspark import SQLContext, SparkContext

from src.modelTraining.deletion import DeletionModel
from src.modelTraining.insertion import InsertionModel
from src.modelTraining.indel import InDelModel

from src.modelTraining.model_factory import log_best, train_model_v2

from hyperopt.pyll.base import scope
from pyspark.sql import SparkSession
import hyperopt
import mlflow
from keras.regularizers import l2, l1

os.environ['PYSPARK_PYTHON'] = "python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


def save_errors(errors, name):
    reg_names = ["l1", "l2"]
    for reg_name, error in zip(reg_names, errors):
        np.save(f"./results/{name}_{reg_name}.npy", error)

def indel_optimization():
    indel_model = InDelModel()
    x_train, x_test, y_train, y_test = indel_model.prepare_data()

    MAX_EVALS = 500
    METRIC = "val_RMSE"
    # Number of experiments to run at once
    PARALLELISM = 2
    
    spark = SparkSession \
        .builder \
        .master("local[4]") \
        .appName("INDEL_PARAM_SEARCH") \
        .getOrCreate()

    space = {
        'num_units': scope.int(hyperopt.hp.quniform('num_units', 2, 64, 2)),
        'num_layers': scope.int(hyperopt.hp.quniform('num_iterations', 0, 10, 1)),
        # The parameters below are cast to int using the scope.int() wrapper
        'kernel_regularizer': hyperopt.hp.choice('kernel_regularizer', ['l1', 'l2']),
        'kernel_weight': 10**hyperopt.hp.quniform('kernel_weight', -10, -1, 1),
        'activation': hyperopt.hp.choice('activation', ['relu', 'sigmoid']),
        'learning_rate': hyperopt.hp.choice('learning_rate', [0.01, 0.001, 0.0001])
    }

    litte_space = {
        'num_units': scope.int(hyperopt.hp.quniform('num_units', 2, 4, 2)),
        'num_layers': scope.int(hyperopt.hp.quniform('num_iterations', 0, 1, 1)),
        # The parameters below are cast to int using the scope.int() wrapper
        'kernel_regularizer': hyperopt.hp.choice('kernel_regularizer', ['l1']),
        'kernel_weight': 10**hyperopt.hp.quniform('kernel_weight', -10, -9, 1),
        'activation': hyperopt.hp.choice('activation', ['relu']),
        'learning_rate': hyperopt.hp.choice('learning_rate', [0.01])
    }



    trials = hyperopt.SparkTrials(parallelism=PARALLELISM, spark_session=spark)
    # trials = hyperopt.SparkTrials()
    train_objective = train_model_v2(x_train, y_train, x_test, y_test, METRIC)

    with mlflow.start_run(experiment_id=1) as run:
        search_run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        print(search_run_id)
        print(experiment_id)

        try:
            hyperopt.fmin(fn=train_objective,
                            space=space,
                            algo=hyperopt.tpe.suggest,
                            max_evals=MAX_EVALS,
                            trials=trials)
            log_best(run, METRIC)
        except BaseException as exp:
            print(f"exception: {exp}")

        


def main():
    faulthandler.enable()
    indel_optimization()
    # del_model = DeletionModel()
    # ins_model = InsertionModel()
    # indel_model = InDelModel()

    # print("---- TRAINING: Indel model ----")
    # indel_errors = indel_model.train_model()
    # save_errors(indel_errors, "indel")
    # print("---- TRAINING: Insertion model ----")
    # ins_errors = ins_model.train_model()
    # save_errors(ins_errors, "insertion")
    # print("---- TRAINING: Deletion model ----")
    # del_errors = del_model.train_model()
    # save_errors(del_errors, "deletion")


if __name__ == "__main__":
    main()
