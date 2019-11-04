import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pydicom
import os
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report
from keras.models import load_model


def process_data(path):
    data = pd.DataFrame([{'path': filepath} for filepath in glob(path)])
    data['file'] = data['path'].map(os.path.basename)
    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data


def count_plot_comparison(feature, data_df, tiff_data, dicom_data):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview data")
    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff files data")
    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom files data")
    plt.show()


def main():
    batch_size = 64
    epochs = 2
    num_classes = 10
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)

    # print(train_X[0,:,:])

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.

    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)

    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])

    train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2,
                                                                  random_state=13)

    print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

    fashion_model = load_model("fashion_model_dropout.h5py")
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

    # fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
    #                       metrics=['accuracy'])

    # fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
    #                                   validation_data=(valid_X, valid_label))

    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)

    fashion_model.save("fashion_model_dropout.h5py")

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

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

    predicted_classes = fashion_model.predict(test_X)
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
    correct = np.where(predicted_classes == test_Y)[0]
    print("Found %d correct labels" % len(correct))

    incorrect = np.where(predicted_classes != test_Y)[0]
    print("Found %d incorrect labels" % len(incorrect))

    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(test_Y, predicted_classes, target_names=target_names))

    data_dir = "../"
    # data_df = pd.read_csv(data_dir + "overview.csv")
    # print("CT Medical images -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
    # data_df.head()
    #
    # print("Number of TIFF images:", len(os.listdir(data_dir + "tiff_images")))
    # tiff_data = pd.DataFrame([{'path': file_path} for file_path in glob('../tiff_images/*.tif')])
    # tiff_data = process_data(data_dir + 'tiff_images/*.tif')
    # # print(tiff_data)
    #
    # print("Number of DICOM files:", len(os.listdir(data_dir + "dicom_dir")))
    # dicom_data = process_data(data_dir + 'dicom_dir/*.dcm')
    #
    # count_plot_comparison('Age', data_df, tiff_data, dicom_data)


if __name__ == "__main__":
    main()
