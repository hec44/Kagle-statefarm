from imutils import paths
import cv2
from sklearn import preprocessing,cross_validation
from sklearn.externals import joblib
import os
def get_train_data(path,img_cols,img_rows,num_imgs):
	i=1
	x_img=[]
	y_labels=[]
	for imagePath in paths.list_images(path):
		i+=1
		img=cv2.imread(imagePath)
		print(imagePath,str(i))
		imagePath=imagePath.split("/")
		y_labels.append(imagePath[1])#la lista de nombres se tiene asi[train,c0,img3012341.jpg]
		res = cv2.resize(img, (img_cols, img_rows))
		res=res.transpose(2,0,1)#Ahora tenemos la matriz como 3x32x32
		x_img.append(res)
		if i==num_imgs:
			return x_img,y_labels

x,y=get_train_data("train/c0",32,32,2000)
x1,y1=get_train_data("train/c1",32,32,2000)
x2,y2=get_train_data("train/c2",32,32,2000)
x3,y3=get_train_data("train/c3",32,32,2000)
x4,y4=get_train_data("train/c4",32,32,2000)
x5,y5=get_train_data("train/c5",32,32,2000)
x6,y6=get_train_data("train/c6",32,32,2000)
x=x+x1+x2+x3+x4+x5+x6
y=y1+y2+y3+y4+y5+y6
data_X,data_X_test,data_Y,data_Y_test=cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)
joblib.dump(data_X, 'dx.pkl')
joblib.dump(data_X_test, 'dxt.pkl')
joblib.dump(data_Y, 'dy.pkl')
joblib.dump(data_Y_test, 'dyt.pkl')

