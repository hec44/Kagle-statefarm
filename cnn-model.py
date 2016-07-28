from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.externals import joblib
import argparse
import h5py

batch_size = 32
nb_classes = 10
nb_epoch = 200

img_channels = 3
def kagle_load_data(path):
    X_train=joblib.load(path+"/dx.pkl")
    X_test=joblib.load(path+"/dxt.pkl")
    Y_train=joblib.load(path+"/dy.pkl")
    Y_test=joblib.load(path+"/dyt.pkl")
    return X_train,X_test,Y_train,Y_test
def load_data(nb_classes,path):
    X_train, X_test,y_train, y_test = kagle_load_data(path)
    print (y_train)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train,X_test,Y_train,Y_test
def create_compile(img_channels,img_rows,img_cols):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

def compile_model(model,X_train,X_test,Y_train,Y_test,batch_size,nb_epoch):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
    model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    return model
def save_model(model,target_path):
    model_json = model.to_json()
    open(target_path+'/model_arch.json', 'w').write(model_json)
    model.save_weights(target_path+'/model_weights.h5', overwrite=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser("modelgenerator.py")
    p.add_argument("data_path",default=None,action="store", help="path to image files")
    p.add_argument("target_path",default=None,action="store", help="target path to model files")
    p.add_argument("-ne","--number_of_epoch",default=200,action="store", help="number_of_epoch")
    opts = p.parse_args()
    batch_size = 32
    nb_classes = 10
    nb_epoch = int(opts.number_of_epoch)
    img_channels = 3
    img_rows=32
    img_cols=32
    X_train,X_test,Y_train,Y_test=load_data(nb_classes,opts.data_path)
    model=create_compile(img_channels,img_rows,img_cols)
    model=compile_model(model,X_train,X_test,Y_train,Y_test,batch_size,nb_epoch)
    save_model(model,opts.target_path)
