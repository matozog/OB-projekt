import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from skimage.io import imread
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ReLU
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.utils.testing import mock_mldata_urlopen
from skimage.transform import resize


def process_data(path):
    data = pd.DataFrame([{'path': filepath} for filepath in glob(path)])
    data['file'] = data['path'].map(os.path.basename)
    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data


def count_plot_comparison(feature, data_df, tiff_data, dicom_data):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview data")
    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff files data")
    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom files data")
    plt.show()


def load_data(tif_data):
    RefDs = imread(tif_data['path'][0])
    IMG_PX_SIZE = 256

    ConstPixelDims = (len(tif_data), IMG_PX_SIZE, IMG_PX_SIZE)

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.dtype)

    listFilesTIF = [path for path in tif_data['path']]
    labels = np.array([label for label in tif_data['Contrast']])

    # loop through all the TIF files
    for filenameDCM in listFilesTIF:
        # read the file
        ds = imread(filenameDCM)
        resized_img = resize(ds, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)

        ArrayDicom[listFilesTIF.index(filenameDCM), :, :] = resized_img
        # show_images(np.expand_dims(ds.pixel_array, axis=0))

    return ArrayDicom, labels


def preprocessing_data(tifSet, labels):
    filters_per_image = 4
    input_shape = 256
    ConstPixelDims = (len(tifSet) * filters_per_image, input_shape, input_shape)
    processedTIFSet = np.zeros(ConstPixelDims)
    processedLabels = np.zeros(len(labels) * filters_per_image, dtype=labels.dtype)
    aug = ImageDataGenerator(
        rotation_range=270,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        data_format="channels_last")

    # dcmSet = dcmSet.reshape(-1, input_shape, input_shape, 1)
    index = 0
    # imageGen = aug.flow(dcmSet)
    # for i in range(len(dcmSet)):
    #     for j in range(filters_per_image):
    #         processedDCMSet[index, :, :] = imageGen
    #         processedLabels[index] = labels[i]
    #         index += 1
    #     show_images(processedDCMSet[index - filters_per_image: index, :, :])

    for i in range(len(tifSet)):
        processedTIFSet[index, :, :] = tifSet[i]
        processedLabels[index] = labels[i]
        index += 1
        processedTIFSet[index, :, :] = np.rot90(tifSet[i])
        processedLabels[index] = labels[i]
        index += 1
        processedTIFSet[index, :, :] = np.rot90(tifSet[i], 2)
        processedLabels[index] = labels[i]
        index += 1
        processedTIFSet[index, :, :] = np.rot90(tifSet[i], 3)
        processedLabels[index] = labels[i]
        index += 1
        # show_images(processedDCMSet[index-filters_per_image: index, :, :])

    return processedTIFSet, processedLabels


def show_images(images):
    plt.figure(figsize=[len(images), len(images)])
    for i in range(len(images)):
        plt.subplot(121)
        plt.imshow(images[i, :, :], cmap='gray')
        plt.show()


def main():
    batch_size = 20
    epochs = 20
    input_shape = 256
    n_classes = 2
    data_dir = "../"
    data_df = pd.read_csv(data_dir + "overview.csv")
    aug = ImageDataGenerator(
        rotation_range=270,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        data_format="channels_last")

    tif_data = process_data(data_dir + 'tiff_images/*.tif')

    # count_plot_comparison('Age', data_df, tiff_data, dicom_data)

    tifSet, labels = load_data(tif_data)

    processedTIFSet, processedLabels = preprocessing_data(tifSet, labels)

    X_train, X_test, y_train, y_test = train_test_split(processedTIFSet, processedLabels, test_size=0.143, shuffle=True,
                                                        random_state=50)

    # plt.figure(figsize=[5, 5])
    # Display the first image in training data
    # plt.subplot(121)
    # plt.imshow(dcmSet[0, :, :], cmap=plt.cm.bone)
    # plt.title("Ground Truth : {}".format(train_Y[0]))

    # Display the first image in testing data
    # plt.subplot(122)
    # plt.imshow(test_X[1, :, :], cmap='gray')
    # plt.title("Ground Truth : {}".format(test_Y[0]))
    # plt.show()

    X_train = X_train.reshape(-1, input_shape, input_shape, 1)
    X_test = X_test.reshape(-1, input_shape, input_shape, 1)

    # print('Training data : ', X_train.shape, y_train.shape)
    #
    # print('Testing data : ', X_test.shape, y_test.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / float(pow(2, 16))
    X_test = X_test / float(pow(2, 16))

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=13)
    # loading model
    # fashion_model = load_model("fashion_model_dropout.h5py")

    first_model = Sequential()
    # first article
    # first_model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(input_shape, input_shape, 1)))
    # first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # first_model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    # first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # first_model.add(Flatten())
    # first_model.add(Dense(150, activation='relu'))
    # first_model.add(Dense(100, activation='relu'))
    # first_model.add(Dense(50, activation='relu'))
    # first_model.add(Dense(2, activation='softmax'))

    # second article
    # first_model.add(Conv2D(12, kernel_size=(7, 7), strides=(1, 1), activation='relu', input_shape=(512, 512, 1), bias_initializer=keras.initializers.Constant(value=0.0)))
    # first_model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # first_model.add(Conv2D(30, kernel_size=(7, 7), strides=(1, 1), activation='relu', bias_initializer=keras.initializers.Constant(value=0.1)))
    # first_model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # first_model.add(Flatten())
    # first_model.add(Dense(500, activation='relu', bias_initializer=keras.initializers.Constant(value=0.1)))
    # first_model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # first_model.add(Dropout(0.5))
    # first_model.add(Dense(2, activation='softmax', bias_initializer=keras.initializers.Constant(value=0.0)))

    # third model
    first_model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(input_shape, input_shape, 1)))
    first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    first_model.add(Flatten())
    first_model.add(Dense(n_classes, activation='softmax'))

    # first article
    # sgd = keras.optimizers.SGD(learning_rate=0.0005, momentum=0.95)
    # adam = keras.optimizers.Adam(learning_rate=0.0005)

    # second article
    # sgd = keras.optimizers.SGD(learning_rate=0.0005, decay=5.5)

    # third model
    adam = keras.optimizers.Adam()

    first_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam,
                        metrics=['accuracy'])

    # first_model.save("first_model.h5py")
    first_model.summary()

    first_train = first_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(X_valid, y_valid))
    # aug.fit(X_train)
    # first_train = first_model.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size),
    #                                         epochs=epochs, verbose=1,
    #                                         validation_data=(X_valid, y_valid))

    test_eval = first_model.evaluate(X_test, y_test_one_hot, verbose=0)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    accuracy = first_train.history['accuracy']
    val_accuracy = first_train.history['val_accuracy']
    loss = first_train.history['loss']
    val_loss = first_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    predicted_classes = first_model.predict(X_test)
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

    correct = np.where(predicted_classes == y_test)[0]
    print("Found %d correct labels" % len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(input_shape, input_shape), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
        plt.tight_layout()
    plt.show()

    incorrect = np.where(predicted_classes != y_test)[0]
    print("Found %d incorrect labels" % len(incorrect))
    for i, incorrect in enumerate(incorrect[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(input_shape, input_shape), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
        plt.tight_layout()

    plt.show()
    target_names = ["Class {}".format(i) for i in range(n_classes)]
    print(classification_report(y_test, predicted_classes, target_names=target_names))

    confusion_matrix_result = confusion_matrix(y_test, predicted_classes)
    print(confusion_matrix_result)


if __name__ == "__main__":
    main()
