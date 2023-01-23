"""The main execution process of the program.

This python file will run the in the style of a pipeline. In particular, code for A1, A2, B1 and B2 will be executed
consecutively.

"""
from A1.infer_cnn import make_datasets as make_datasets_a1
from A1.infer_cnn import train
from A1.infer_cnn import run_saved_model as run_cnn

from A2.infer_svm import infer
from A2.infer_svm import run_trained_model as run_svm

from B1.random_forest import run_saved_model as run_rf

from B2.infer_resnet import run_saved_model as run_resnet
from B2.infer_resnet import make_datasets as make_datasets_b2


def main():
    """The entrance of the program. Run code for TaskA1, TaskA2, TaskB1, TaskB2 consecutively.
    """
    # A1 section
    print('A1 task running...')
    # 1. make datasets
    datasets, train_ds, val_ds, test_ds = make_datasets_a1()
    # 2. train the model, set cv_option to True if cross validation is wanted
    train(datasets, train_ds, val_ds, test_ds, cv_option=False)
    # 3. test the model
    run_cnn(test_ds)
    print('A1 task ending...')

    # A2 section
    print('A2 task running...')
    # 1. train the model
    infer()
    # 2. test the model
    run_svm()
    print('A2 task ending...')

    # B1 section
    print('B1 task running...')
    # 1. train the model
    infer()
    # 2. test the model
    run_rf()
    print('B1 task ending...')

    # B2 section
    print('B2 task running...')
    # 1. make datasets
    dataset, train_ds, val_ds, test_ds = make_datasets_b2()
    # 2. train the model, set cv_option to True if cross validation is wanted
    train(dataset, train_ds, val_ds, test_ds, cv_option=False)
    # 3. test the model
    run_resnet(test_ds)
    print('B2 task ending...')


if __name__ == '__main__':
    main()
