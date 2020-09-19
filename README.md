# underwater-gesture-recognition

This repository contains the codes used in the paper "Underwater Gesture Recognition Using Classical Computer Vision and Deep Learning Techniques". The codes are divided into three groups: one for each model used in the paper.

### 1. Bag of Visual Words (BOVW):
- Scripts (adopted from Kushal Vyas’ implementation of Bag of Visual Words: https://github.com/kushalvyas/Bag-of-Visual-Words-Python)  
	-```Bag.py``` – main script for training and testing  
	-```helper.py``` – contains helper scripts for training and testing  
- To test:  
	1. Put train images in ```./images/train folder```; this is needed to get the classes to be used in testing
	2. Put test images in ```./images/test folder```
	3. Put ```kmeans_cluster_3.sav```, ```mega_histogram_3.pkl```, ```svm_train_3.pkl```, ```and vstack_3.pkl``` in the same directory as Bag.py and helpers.py   
    	4. run ```python Bag.py```  
- Links:
	-Link to train images: https://drive.google.com/open?id=1JCXaT9PSnyJ4KgTzlumMrmwALVt-2V_t
	-Link to test images: https://drive.google.com/open?id=1ducpYZnC9dQznG90HbiyZSX3CYCR8UbR  
	-Link to pretrained weights: https://drive.google.com/open?id=1hSaZwRpbtOqFYep7Z4jW2kkKizEHoVqO  
- Dependencies:  
    o	OpenCV 3.4.2.17 (OpenCV-contrib-python==3.4.2.17), scikit-learn 0.20.3  

### 2. Histogram of Gradients (HOG):  
  •	Scripts:  
    o	train.py  
    o	trim_hnm.py   
    o	testing.py  
    o	visualize_predictions.py  
  •	To test:  
    o	Put test images in ./images/test folder   
    Link to test images: https://drive.google.com/open?id=1ducpYZnC9dQznG90HbiyZSX3CYCR8UbR  
    o	Put hog_svm.joblib pre-trained model in ./models folder   
    Link to pretrained weights: https://drive.google.com/open?id=1pirGWIkZqWXBNSwQKuTdLpmMKowYrOUK  
    o	Create ./results/ folder, where the csv containing the target classes and predictions will be saved  
    o	Run testing.py  
  •	To visualize sample correct and incorrect predictions per class:  
    o	Create ./visualization/ folder, which will contain the sample images with predictions  
    o	Run visualize_predictions.py  
  •	Dependencies:  
    o	OpenCV 4.1.0.25 (opencv-contrib-python==4.1.0.25), scikit-learn 0.21.0  

### 3. ResNet50-CNN  
  •	Scripts:  
    o	data_augmentation.py  
    o	classifier_withtest.py  
  •	To test:  
    •Place the test in ./images/test folder  
    o Put the pre-trained model resnet50_CNN_caddy.pickle in the same directory as classifier_withtest.py  
    Link to pretrained weights: https://drive.google.com/file/d/1EQxRwaFTZVlmVlTZDVx5Y1fcLAxjL6Qj/view?usp=sharing  
    o Run classifier_withtest.py  
   •	Dependencies:  
    o pytorch 0.4.0  
    o Cuda 9.0  
		
    
    
  
   
  
