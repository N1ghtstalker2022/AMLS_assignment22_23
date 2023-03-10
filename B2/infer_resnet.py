"""Solve B2 task

This python file contains the code for training, validation and testing the resnet-34 model.

"""
import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

AUTOTUNE = tf.data.AUTOTUNE
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
batch_size = 32
epochs = 10
class_dic = {0: 'eye_color_1', 1: 'eye_color_2', 2: 'eye_color_3', 3: 'eye_color_4', 4: 'eye_color_5'}
img_width = 50
img_height = 50
img_shape = None


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def resnet_34(input_shape, classes=10):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 255, input_shape=input_shape)(x_input),
    x = x[0]
    x = tf.keras.layers.Resizing(img_width, img_height)(x),
    x = x[0]
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model


def cross_validation(dataset):
    train_dataset = [[], []]
    for elem in dataset:
        train_dataset[0].append(elem[0].numpy())
        train_dataset[1].append(elem[1].numpy())

    # Cross-validation process -----------------------------------------------------------------------------------------
    possible_learning_rates = [1e-2, 1e-3, 1e-4]
    folds_num = 3
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
            interim_model = resnet_34(img_shape, classes=5)

            # Compile the model
            interim_model.compile(loss=loss_function,
                                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
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
    return best_learning_rate


def train(dataset, train_ds, val_ds, test_ds, cv_option=False):
    # configure the dataset for performance.
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    for img, _ in train_ds.take(1):
        global img_shape
        img_shape = img.shape[1:]

    # cross validation
    if cv_option:
        best_learning_rate = cross_validation(dataset)

    else:
        best_learning_rate = 1e-4

    model = resnet_34(img_shape, classes=5)

    # Compile the model ------------------------------------------------------------------------------------------------
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                  loss=loss_function,
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )

    # view the model summary
    model.summary()

    # Visualizing training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(loss)
    print(acc[0])

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
    if cv_option:
        plt.savefig('./training_results.png')
    else:
        plt.savefig('./training_no_cv_results.png')
    plt.close()

    # with k-fold
    model.save('./resnet_model')


def make_datasets():
    # load image
    dataset, dataset_size = make_dataset('../Datasets/dataset_AMLS_22-23/cartoon_set/img',
                                         '../Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv')

    test_set, _ = make_dataset('../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/img',
                               '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv')

    train_size = int(0.8 * dataset_size)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

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

    return dataset, train_ds, val_ds, test_ds


def get_filenames_from_folder(folder):
    filenames = []
    image_file = os.listdir(folder)  # your directory path
    img_num = len(image_file)
    for i in range(img_num):
        filename = str(i) + '.png'
        filenames.append(os.path.join(folder, filename))
    return filenames


def make_dataset(image_path, label_path):
    # load image
    labels_df = pd.read_csv(label_path, sep='\t')
    labels_df = labels_df.drop(labels_df.columns[0], axis=1)
    labels = labels_df.loc[:, 'eye_color'].tolist()

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
        image = tf.cast(image_decoded, tf.float32)
        return image, label

    # dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    # dataset = dataset.map(lambda x: x + 1)

    dataset = dataset.map(_parse_function)
    return dataset, dataset_size


def run_b2():
    dataset, train_ds, val_ds, test_ds = make_datasets()
    train(dataset, train_ds, val_ds, test_ds, cv_option=False)


def run_saved_model(test_ds):
    saved_model = tf.keras.models.load_model('./resnet_model')

    # Check its architecture
    saved_model.summary()

    # Predict
    test_loss = saved_model.evaluate(test_ds)
    print(f'{test_loss}')


if __name__ == '__main__':
    dataset, train_ds, val_ds, test_ds = make_datasets()
    run_saved_model(test_ds)
