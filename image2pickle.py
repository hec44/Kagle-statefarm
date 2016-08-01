from imutils import paths
import cv2
from sklearn import preprocessing,cross_validation
from sklearn.externals import joblib
import os
import argparse
import numpy as np
def get_train_data(path,img_cols,img_rows,num_imgs=-1):
	print path
	i=1
	x_img=[]
	y_labels=[]
	if num_imgs!="-1":
		for imagePath in paths.list_images(path):
			i+=1
			print(imagePath,str(i))
			img=cv2.imread(imagePath)

			imagePath=imagePath.split("/")
			y_labels.append(int(imagePath[1].replace("c","")))#once we split, lists are like this
										#[train,c0,img3012341.jpg]
										#we want only integers for CNN model
			res = cv2.resize(img, (img_cols, img_rows))
			res=res.transpose(2,0,1)#Size is now 3x32x32
			x_img.append(res)
			if i==num_imgs:
				return x_img,y_labels
			print i


	else:
		print("HOLAAA")
		for imagePath in paths.list_images(path):
			i+=1
			print(imagePath,str(i))
			img=cv2.imread(imagePath)

			imagePath=imagePath.split("/")
			y_labels.append(int(imagePath[1].replace("c","")))#once we split, lists are like this
										#[train,c0,img3012341.jpg]
										#we want only integers for CNN model
			res = cv2.resize(img, (img_cols, img_rows))
			res=res.transpose(2,0,1)#Size is now 3x32x32
			x_img.append(res)

		return x_img,y_labels
def get_train_folders(root_path,img_cols,img_rows,num_imgs=-1):
	data_x=[]
	data_y=[]
	for path in os.listdir(root_path):
		try:
			aux_x,aux_y=get_train_data(root_path+"/"+path,img_cols,img_rows,num_imgs)
		except TypeError as e:
			print "AAAAH"
		data_x=data_x+aux_x
		data_y=data_y+aux_y
	data_X,data_X_test,data_Y,data_Y_test=cross_validation.train_test_split(data_x, data_y, test_size=0.2, random_state=42)
	return data_X,data_X_test,data_Y,data_Y_test
def img2pkl(root_path,target_path,img_cols,img_rows,num_imgs=-1):
	data_X,data_X_test,data_Y,data_Y_test=get_train_folders(root_path,img_cols,img_rows,num_imgs)
	joblib.dump(np.asarray(data_X), target_path+'/dx.pkl')
	joblib.dump(np.asarray(data_X_test), target_path+'/dxt.pkl')
	joblib.dump(np.asarray(data_Y), target_path+'/dy.pkl')
	joblib.dump(np.asarray(data_Y_test), target_path+'/dyt.pkl')
if __name__ == '__main__':
	p = argparse.ArgumentParser("image2pickle.py")
	p.add_argument("root_path",default=None,action="store", help="path to image files")
	p.add_argument("target_path",default=None,action="store", help="target path to pickle files")
	p.add_argument("-ir","--img_rows",default=32,action="store", help="image rows")
	p.add_argument("-ic","--img_cols",default=32,action="store", help="image columns")
	p.add_argument("-ni","--num_imgs",default=2000,action="store", help="number of images to open")
	opts = p.parse_args()
	img2pkl(opts.root_path,opts.target_path,opts.img_cols,opts.img_rows,opts.num_imgs)
