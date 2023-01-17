import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import svm, preprocessing
import cv2
import random
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, learning_curve
from sklearn.svm import SVC
from joblib import dump, load


def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
            filtered_cv_results["mean_test_precision"],
            filtered_cv_results["std_test_precision"],
            filtered_cv_results["mean_test_recall"],
            filtered_cv_results["std_test_recall"],
            filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()


def refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # print the info about the grid-search for the different scores
    precision_threshold = 0.7

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
        ]

    print(f"Models with a precision higher than {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
        ]
    print(
        "Out of the previously selected high precision models, we keep all the\n"
        "the models within one standard deviation of the highest recall model:"
    )
    print_dataframe(high_recall_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on precision and recall.\n"
        "Its scoring time is:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index


def adjust_dataset(dataset):
    dataset_set_adjust = [[], []]
    # print(dataset[0][0].flatten())
    for data in dataset:
        dataset_set_adjust[0].append(data[0].flatten())
        dataset_set_adjust[1].append(data[1])
    return dataset_set_adjust


def infer():
    # load image
    dataset, dataset_size = make_dataset('../Datasets/dataset_AMLS_22-23/celeba/img',
                                         '../Datasets/dataset_AMLS_22-23/celeba/labels.csv')

    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/celeba_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv')

    train_size = int(0.8 * dataset_size)

    train_set_indices = random.sample(range(dataset_size), train_size)

    # adjust data format for easy input
    dataset = adjust_dataset(dataset)
    test_set = adjust_dataset(test_set)

    train_set = [[], []]
    val_set = [[], []]
    for index in range(dataset_size):
        train_set[0].append(dataset[0][index])
        train_set[1].append(dataset[1][index])

    # define input
    train_data = train_set[0]
    train_label = train_set[1]
    test_data = test_set[0]
    test_label = test_set[1]

    # used for plot samples
    class_dic = {1: 'smiling', 0: 'not smiling'}

    # plot some samples
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        cur_img = train_data[i].reshape(50, 50, 3).astype("uint8")
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        plt.imshow(cur_img)
        plt.title(class_dic[train_label[i]])
        plt.axis("off")
    plt.savefig('./sample.png')
    plt.close()

    # preprocessing, scaling
    scaler_train = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler_train.transform(train_data)
    scaler_test = preprocessing.StandardScaler().fit(test_data)
    test_data = scaler_test.transform(test_data)

    scores = ["precision", "recall"]

    # Exhaustive Grid Search to select best estimators
    # hyperparameter set
    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]},
        {"kernel": ["linear"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]},
    ]

    grid_search = GridSearchCV(
        SVC(), tuned_parameters, scoring=scores, cv=5, refit=refit_strategy, verbose=1
    )

    # best_model = SVC(C=1, kernel='rbf', verbose=1)

    grid_search.fit(train_data, train_label)

    best_model = grid_search.best_estimator_

    print(best_model)
    #
    # size, train_scores, valid_scores = learning_curve(best_model, train_data, train_label, train_sizes=np.linspace(0.1, 1.0, 10))
    # print(size)
    # plt.plot(size, np.mean(train_scores, axis=1), 'r--', label='Training Accuracy')
    # plt.plot(size, np.mean(valid_scores, axis=1), 'b--', label='Validation Accuracy')
    # # plt.xlabel('x label')
    # # plt.ylabel('y label')
    # plt.title("Training and Validation Accuracy")
    # plt.legend()
    # plt.savefig('./learning_curve.png')

    # test svm model
    print(grid_search.best_params_)

    # get a glance of the model performance
    # plot some samples
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        cur_img = test_data[i].reshape(50, 50, 3).astype("uint8")
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        plt.imshow(cur_img)
        plt.title(class_dic[train_label[i]])
        plt.axis("off")
    plt.savefig('./prediction_sample.png')
    plt.close()
    # save the model
    dump(best_model, './svm_model.joblib')


def make_dataset(image_path, label_path):
    # load image
    labels_df = pd.read_csv(label_path, sep='\t')
    labels_df = labels_df.drop(labels_df.columns[0], axis=1)
    labels = labels_df.loc[:, 'smiling'].tolist()
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 0

    # Convert to tensorflow datasets
    # step 1
    filenames = get_filenames_from_folder(image_path)
    dataset_size = len(filenames)

    # step 2: create a dataset returning slices of `filenames`

    # step 3: parse every image in the dataset using `map`

    def _parse_function(filename, label):
        image = cv2.imread(filename)
        image_decoded = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)

        return image_decoded, label

    # dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    # dataset = dataset.map(lambda x: x + 1)

    dataset = map(_parse_function, filenames, labels)
    return list(dataset), dataset_size


def get_filenames_from_folder(folder):
    print(folder)
    filenames = []
    image_file = os.listdir(folder)  # your directory path
    img_num = len(image_file)
    for i in range(img_num):
        filename = str(i) + '.jpg'
        filenames.append(os.path.join(folder, filename))
    return filenames


def run_trained_model():
    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/celeba_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv')

    test_set = adjust_dataset(test_set)

    test_data = test_set[0]

    test_label = test_set[1]

    model = load('svm_model.joblib')

    predict_label = model.predict(test_data)

    print(accuracy_score(test_label, predict_label))


if __name__ == '__main__':
    run_trained_model()
