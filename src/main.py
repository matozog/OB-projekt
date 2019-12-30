import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pydicom
import os
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.metrics import classification_report

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


def load_DCM_data(dicom_data):
    # Get ref file
    RefDs = pydicom.dcmread(dicom_data['path'][0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (len(dicom_data), int(RefDs.Rows), int(RefDs.Columns))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    # y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    # z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    listFilesDCM = [path for path in dicom_data['path']]
    labels = np.array([label for label in dicom_data['Contrast']])

    # loop through all the DICOM files
    for filenameDCM in listFilesDCM:
        # read the file
        ds = pydicom.dcmread(filenameDCM)
        # store the raw image data
        ArrayDicom[listFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

    return ArrayDicom, labels


def main():
    batch_size = 10
    epochs = 10
    num_classes = 10
    n_classes = 2
    data_dir = "../"
    data_df = pd.read_csv(data_dir + "overview.csv")
    print("CT Medical images -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
    data_df.head()

    print("Number of TIFF images:", len(os.listdir(data_dir + "tiff_images")))
    tiff_data = pd.DataFrame([{'path': file_path} for file_path in glob('../tiff_images/*.tif')])
    tiff_data = process_data(data_dir + 'tiff_images/*.tif')
    # print(tiff_data)

    print("Number of DICOM files:", len(os.listdir(data_dir + "dicom_dir")))
    dicom_data = process_data(data_dir + 'dicom_dir/*.dcm')

    # count_plot_comparison('Age', data_df, tiff_data, dicom_data)

    # print(dicom_data.shape)
    dcmSet, labels = load_DCM_data(dicom_data)

    # transpozycja x z y
    # for i in range(len(dcmSet)):
    #     dcmSet[i] = dcmSet[i].transpose()

    # print(dcmSet)
    # print(dcmSet.shape)
    # print(len(dcmSet))
    # print(labels)

    # example from https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(dcmSet, labels, test_size=0.143, shuffle=True, random_state=50)

    plt.figure(figsize=[5, 5])
    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(dcmSet[8, :, :], cmap=plt.cm.bone)
    # plt.title("Ground Truth : {}".format(train_Y[0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_X[1, :, :], cmap='gray')
    plt.title("Ground Truth : {}".format(test_Y[0]))
    plt.show()

    # print(dcmSet.shape[0])
    # print(labels.shape[0])

    # print(X_test)
    # print(y_test)

    # print(test_X.shape)
    # print(test_X)

    # reshaping data
    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)

    X_train = X_train.reshape(-1, 512, 512, 1)
    X_test = X_test.reshape(-1, 512, 512, 1)

    # print('Training data shape : ', train_X.shape, train_Y.shape)
    #
    # print('Testing data shape : ', test_X.shape, test_Y.shape)
    #
    # print('Training data : ', X_train.shape, y_train.shape)
    #
    # print('Testing data : ', X_test.shape, y_test.shape)

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / float(pow(2, 16))
    X_test = X_test / float(pow(2, 16))

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=13)
    # print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    # print(y_train_one_hot[0])
    # print(test_X)
    # print(X_test)

    # train_Y_one_hot = to_categorical(train_Y)
    # test_Y_one_hot = to_categorical(test_Y)
    #
    # print('Original label:', train_Y[0])
    # print('After conversion to one-hot:', train_Y_one_hot[0])
    #
    # train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2,
    #                                                               random_state=13)
    #
    # print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

    # loading model
    # fashion_model = load_model("fashion_model_dropout.h5py")

    # building new model
    # fashion_model = Sequential()
    # fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    # fashion_model.add(Dropout(0.25))
    # fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # fashion_model.add(Dropout(0.25))
    # fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # fashion_model.add(Dropout(0.4))
    # fashion_model.add(Flatten())
    # fashion_model.add(Dense(128, activation='linear'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(Dropout(0.3))
    # fashion_model.add(Dense(num_classes, activation='softmax'))

    first_model = Sequential()
    first_model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1), activation='linear', input_shape=(512, 512, 1)))
    first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    first_model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='linear'))
    first_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    first_model.add(Flatten())
    first_model.add(Dense(150, activation='relu'))
    first_model.add(Dense(100, activation='relu'))
    first_model.add(Dense(50, activation='relu'))
    first_model.add(Dense(2, activation='softmax'))

    first_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])
    # first_model.save("first_model.h5py")
    first_model.summary()

    first_train = first_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                      validation_data=(X_valid, y_valid))

    test_eval = first_model.evaluate(X_test, y_test_one_hot, verbose=0)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    predicted_classes = first_model.predict(X_test)
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

    correct = np.where(predicted_classes == y_test)[0]
    print("Found %d correct labels" % len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(512, 512), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
        plt.tight_layout()
    plt.show()

    incorrect = np.where(predicted_classes != y_test)[0]
    print("Found %d incorrect labels" % len(incorrect))
    for i, incorrect in enumerate(incorrect[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(512, 512), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
        plt.tight_layout()

    plt.show()
    target_names = ["Class {}".format(i) for i in range(n_classes)]
    print(classification_report(y_test, predicted_classes, target_names=target_names))

    # fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
    #                       metrics=['accuracy'])

    # fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
    #                                   validation_data=(valid_X, valid_label))

    # evaluate model
    # test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    #
    # fashion_model.save("fashion_model_dropout.h5py")
    #
    # print('Test loss:', test_eval[0])
    # print('Test accuracy:', test_eval[1])

    # accuracy = fashion_train.history['acc']
    # val_accuracy = fashion_train.history['val_acc']
    # loss = fashion_train.history['loss']
    # val_loss = fashion_train.history['val_loss']
    # epochs = range(len(accuracy))
    # plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    # plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

    # predicted_classes = fashion_model.predict(test_X)
    # predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
    # correct = np.where(predicted_classes == test_Y)[0]
    # print("Found %d correct labels" % len(correct))
    #
    # incorrect = np.where(predicted_classes != test_Y)[0]
    # print("Found %d incorrect labels" % len(incorrect))
    #
    # target_names = ["Class {}".format(i) for i in range(num_classes)]
    # print(classification_report(test_Y, predicted_classes, target_names=target_names))


if __name__ == "__main__":
    main()
