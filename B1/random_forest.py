"""Solve B1 task

This python file contains the code for training, validation and testing the random forest model.

"""
import os
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from matplotlib import pyplot as plt


def make_dataset(image_path, label_path):
    # load image
    labels_df = pd.read_csv(label_path, sep='\t')
    labels_df = labels_df.drop(labels_df.columns[0], axis=1)
    labels = labels_df.loc[:, 'face_shape'].tolist()

    # Convert to tensorflow datasets
    # step 1
    filenames = tf.constant(get_filenames_from_folder(image_path))
    dataset_size = len(filenames)
    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # step 3: parse every image in the dataset using `map`

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [50, 50])
        image = tf.cast(tf.reshape(image_resized, [-1]), tf.float32)
        return image, label

    # dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    # dataset = dataset.map(lambda x: x + 1)

    dataset = dataset.map(_parse_function)
    return dataset, dataset_size


def get_filenames_from_folder(folder):
    print(folder)
    filenames = []
    image_file = os.listdir(folder)  # your directory path
    img_num = len(image_file)
    for i in range(img_num):
        filename = str(i) + '.png'
        filenames.append(os.path.join(folder, filename))
    return filenames


def infer():
    # load image
    train_set, dataset_size = make_dataset('../Datasets/dataset_AMLS_22-23/cartoon_set/img',
                                           '../Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv')

    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv')


    # used for plot samples
    class_dic = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4', 4: 'type5'}

    # img_count = 1
    # for image, label in train_set.take(9):
    #     print("...")
    #     label = label.numpy()
    #     ax = plt.subplot(3, 3, img_count)
    #     img_count = img_count + 1
    #     plt.imshow(image.numpy().astype("uint8"))
    #     plt.title(class_dic[label])
    #     plt.axis("off")
    # plt.savefig('./sample_data.png')
    # plt.close()

    # Add batch operations
    batch_size = 256
    train_set = train_set.batch(batch_size)
    test_set = test_set.batch(batch_size)

    # Train a model with default hyperparameters
    model = tfdf.keras.RandomForestModel(verbose=0)

    model.fit(train_set)

    logs = model.make_inspector().training_logs()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")
    plt.savefig('./training_results.png')
    plt.close()

    # Summary of the model structure.
    model.summary()

    # Evaluate the model.
    test_loss = model.evaluate(test_set)
    print(f'{test_loss}')


    # Save the model.
    model.save("./rf_model")

def run_saved_model():
    saved_model = tf.keras.models.load_model('./rf_model')
    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv')

    test_set = adjust_dataset(test_set)
    test_ds = test_set[0]

    test_label = test_set[1]
    # Predict
    test_loss = saved_model.evaluate(test_ds)
    print(f'{test_loss}')

def adjust_dataset(dataset):
    dataset_set_adjust = [[], []]
    # print(dataset[0][0].flatten())
    for data in dataset:
        dataset_set_adjust[0].append(data[0].flatten())
        dataset_set_adjust[1].append(data[1])
    return dataset_set_adjust


if __name__ == '__main__':
    # test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/img',
    #                            '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv')
    # batch_size = 256
    # test_set = test_set.batch(batch_size)
    # run_saved_model(test_set)
    infer()
