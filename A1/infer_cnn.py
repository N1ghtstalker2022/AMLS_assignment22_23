import os
import sys

import numpy
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras import datasets, layers, models, Sequential
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from tensorflow.python.data import Dataset, make_one_shot_iterator

AUTOTUNE = tf.data.AUTOTUNE
# change to
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def infer():
    # load image
    dataset, dataset_size = make_dataset('../Datasets/dataset_AMLS_22-23/celeba/img',
                                         '../Datasets/dataset_AMLS_22-23/celeba/labels.csv')

    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/celeba_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/celeba_test/labels.csv')
    # used for plot
    class_dic = {1: 'male', 0: 'female'}
    label_num = len(class_dic)

    train_size = int(0.8 * dataset_size)
    # val_size = dataset_size - train_size
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    train_dataset = [[], []]
    for elem in train_ds:
        train_dataset[0].append(elem[0].numpy())
        train_dataset[1].append(elem[1].numpy())

    batch_size = 32
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_set.batch(batch_size)

    for images, labels in train_ds.take(1):
        print("...")
        labels = labels.numpy()
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_dic[labels[i]])
            plt.axis("off")
        plt.savefig('./sample_data.png')
        plt.close()

    # configure the dataset for performance.

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # define fixed image size for input
    img_height = 250
    img_width = 250
    # create the model, CNN
    model = Sequential([
        layers.Resizing(img_height, img_width),
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(label_num)
    ])

    # Cross-validation process -----------------------------------------------------------------------------------------
    best_learning_rate = 1e-3
    possible_learning_rates = [1e-2, 1e-3, 1e-4]
    folds_num = 10
    # Define the K-fold Cross Validator
    k_fold = KFold(n_splits=folds_num, shuffle=True)
    average_error_list = []

    # K-fold Cross Validation model evaluation
    for learning_rate in possible_learning_rates:
        fold_no = 1
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        for train_index, test_index in k_fold.split(train_dataset[0], train_dataset[1]):
            # Define the model architecture
            interim_model = Sequential([
                layers.Resizing(img_height, img_width),
                layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(label_num)
            ])

            # Compile the model
            interim_model.compile(loss=loss_function,
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                  metrics=['accuracy'])

            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')

            # Fit data to model
            inputs = []
            label_for_inputs = []

            for index in train_index:
                inputs.append(train_dataset[0][index])
                label_for_inputs.append(train_dataset[1][index])

            test_inputs = []
            test_label_for_inputs = []

            for index in test_index:
                test_inputs.append(train_dataset[0][index])
                test_label_for_inputs.append(train_dataset[1][index])
            history = interim_model.fit(np.array(inputs), np.array(label_for_inputs),
                                        batch_size=batch_size,
                                        epochs=1,
                                        verbose=1)

            # Generate generalization metrics

            scores = interim_model.evaluate(np.array(test_inputs), np.array(test_label_for_inputs), verbose=0)
            print(
                f'Score for fold {fold_no}: {interim_model.metrics_names[0]} of {scores[0]}; '
                f'{interim_model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1
        average_error_list.append(np.mean(loss_per_fold))

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print(f'Current learning_rate is: {learning_rate}')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

    min_index = 0
    min_error = sys.maxsize
    for i in range(len(possible_learning_rates)):
        if average_error_list[i] < min_error:
            min_index = i
            min_error = average_error_list[i]

    best_learning_rate = possible_learning_rates[min_index]

    print(f'Best learning rate learned is {best_learning_rate}')
    # Compile the model ------------------------------------------------------------------------------------------------
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                  loss=loss_function,
                  metrics=['accuracy'])

    # Train the model
    epochs = 1
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # view the model summary
    model.summary()

    # Visualizing training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_results.png')
    plt.close()

    # predict
    test_loss = model.evaluate(test_ds)
    print(f'{test_loss}')
    #
    # # implement dropout ------------------------------------------------------------------------------------------------
    # model_dropout = Sequential([
    #     layers.Rescaling(1. / 255),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(label_num, name="outputs")
    # ])
    # model.compile(optimizer='adam',
    #               loss=loss_function,
    #               metrics=['accuracy'])
    # epochs = 15
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )
    # model.summary()
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs_range = range(epochs)
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.savefig('training_results_dropout.png')
    # plt.close()

    print("!!!")


def make_dataset(image_path, label_path):
    # load image
    labels_df = pd.read_csv(label_path, sep='\t')
    labels_df = labels_df.drop(labels_df.columns[0], axis=1)
    labels = labels_df.loc[:, 'gender'].tolist()
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = 0

    # Convert to tensorflow datasets
    # step 1
    filenames = tf.constant(get_filenames_from_folder(image_path))
    dataset_size = len(filenames)
    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # step 3: parse every image in the dataset using `map`

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
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
        filename = str(i) + '.jpg'
        filenames.append(os.path.join(folder, filename))
    return filenames


if __name__ == '__main__':
    infer()
