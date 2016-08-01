from keras.models import model_from_json
import numpy as np
import cv2
from keras.optimizers import SGD
import scipy.misc
import pandas
import csv

def get_test_data(path,img_cols,img_rows,model):

    colnames=["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
    data=pandas.read_csv("sample_submission.csv", names=colnames)
    img_names=data.img.tolist()[1:]
    images=[]
    counter=0
    for img in img_names:
        pic=scipy.misc.imresize(scipy.misc.imread("test/"+img),(img_cols,img_rows))
        pic=np.transpose(pic,(2, 0, 1))
        pic=pic.astype('float32')
        images.append(pic)
        counter=counter+1
        print counter
    return ((np.array(images) / 255),img_names)

def predictions2csv(model, data,csv_name,img_names):
    predictions= model.predict_proba(data)
    csvfile=open(csv_name, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"])
    aux=0
    for i in range(len(predictions)):
        writer.writerow([img_names[aux]]+list(predictions[i]))
        aux=aux+1
    return predictions

def load_model(path):
   model = model_from_json(open(path+"/model_arch.json").read())
   model.load_weights(path+"/model_weights.h5")
   sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'])
   return model

if __name__ == '__main__':
    model=load_model("model")
    data = get_test_data("test",32,32,model)
    predictions2csv(model,data[0],"final.csv",data[1])
