import numpy as np

from src.deletion import DeletionModel
from src.insertion import InsertionModel
from src.indel import InDelModel


def save_errors(errors, name):
    reg_names = ["l1", "l2"]
    for reg_name, error in zip(reg_names, errors):
        np.save(f"./results/{name}_{reg_name}.npy", error)


def main():
    del_model = DeletionModel()
    ins_model = InsertionModel()
    indel_model = InDelModel()

    print("---- TRAINING: Indel model ----")
    indel_errors = indel_model.train_model()
    save_errors(indel_errors, "indel")
    print("---- TRAINING: Insertion model ----")
    ins_errors = ins_model.train_model()
    save_errors(ins_errors, "insertion")
    print("---- TRAINING: Deletion model ----")
    del_errors = del_model.train_model()
    save_errors(del_errors, "deletion")


if __name__ == "__main__":
    main()
