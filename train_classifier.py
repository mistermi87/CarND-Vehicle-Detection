# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:46:05 2018

@author: mstei
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

'''
############ HELPER FUNCTIONS ################
'''
from helper_func import *

'''
############ TRAIN CLASSIFIER ################
'''
#Parameters color only
colorspace_c='YCrCb' #colorspace for the Spatial color distribution (histogram is always on bgr)
spatial = 16 #resize of the images that are taken for spatial color distribution
histbin = 32 #histogram bins for color distribution
hist_range=(0, 256) #range in which the histogram regards the color

#Parameters HOG
colorspace_h='YCrCb' # color space for hog_image
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

#Create a list of car and not-car images

cars = glob.glob('training data\\vehicles/**/*.jpeg')
cars.append(glob.glob('training data\\vehicles/**/*.png'))
cars=cars[0]

notcars = glob.glob('training data\\non-vehicles/*/*.jpeg')
notcars.append(glob.glob('training data\\non-vehicles/*/*.png'))
notcars=notcars[0]

        
car_features = extract_features(cars, colorspace_c, (spatial, spatial),
                        histbin, hist_range,colorspace_h , orient, 
                        pix_per_cell, cell_per_block, hog_channel)[0]
notcar_features, len_features = extract_features(notcars, colorspace_c, (spatial, spatial),
                        histbin, hist_range, colorspace_h, orient, 
                        pix_per_cell, cell_per_block, hog_channel)




X = np.vstack((car_features, notcar_features)).astype(np.float64) 
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))                   


## =============================================================================
#car_ind = np.random.randint(0, len(cars))
## Plot an example of raw and scaled features
#fig = plt.figure(figsize=(12,4))
#plt.subplot(131)
#plt.imshow(mpimg.imread(cars[car_ind]))
#plt.title('Original Image')
#plt.subplot(132)
#plt.plot(X[car_ind])
#plt.title('Raw Features')
##plt.subplot(133)
##plt.plot(scaled_X[car_ind])
##plt.title('Normalized Features')
##fig.tight_layout()
#     
# #evaluate
# # Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)
#    
## Fit a per-column scaler
#X_scaler_train = StandardScaler().fit(X_train)
## Apply the scaler to X
#X_train = X_scaler_train.transform(X_train)
#X_test = X_scaler_train.transform(X_test)
#print('Feature vector length:', len(X_train[0]))
## Use a linear SVC 
#clf1 = LinearSVC(C=1)
#clf1.fit(X_train, y_train)
## Check the score of the SVC
#p1=clf1.predict(X_test)
#pp1=clf1.decision_function(X_test)
#print('Test Accuracy of SVC = ', round(1.0-np.sum(np.abs(p1- y_test))/len(y_test), 4))
##Use gaussianNB
#clf2=RandomForestClassifier(n_estimators=10)
#clf2.fit(X_train[:,0:len_features[0]+len_features[1]-1], y_train)
#p2=clf2.predict(X_test[:,0:len_features[0]+len_features[1]-1])
#pp2=clf2.predict_proba(X_test[:,0:len_features[0]+len_features[1]-1])[:,1]
#print('Test Accuracy of Rand Forrest = ', round(1.0-np.sum(np.abs(p2- y_test))/len(y_test), 4))
#
#ppt=pp1-0.5+pp2
#pp=np.zeros_like(ppt)
#pp[ppt>0]=1
#print('Test Accuracy of Combination = ', round(1.0-np.sum(np.abs(pp- y_test))/len(y_test), 4))

#trees  = [5, 10, 30, 50, 100]
#param_grid = {'n_estimators ': trees}
#grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
#grid_search.fit(X_train, y_train)
#print(grid_search.best_params_)


# =============================================================================

X, y = shuffle(X, y, random_state=0)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

clf1=RandomForestClassifier(n_estimators=10)
clf1.fit(scaled_X[:,0:len_features[0]+len_features[1]-1], y)

clf2 = LinearSVC(C=0.1)
clf2.fit(scaled_X, y)
 
dict_to_save = {colorspace_c: colorspace_c,\
        "spatial": spatial,\
        "histbin":histbin,\
        "hist_range":hist_range,\
        "colorspace_h":colorspace_h,\
        "orient":orient,\
        "pix_per_cell":pix_per_cell,\
        "cell_per_block":cell_per_block,\
        "hog_channel":hog_channel,\
        "X_scaler":X_scaler,\
        "svc":clf2,\
        "rand_forr":clf1,\
        "colorspace_c":colorspace_c,\
        "colorspace_h":colorspace_h,
        "len_features": len_features}
p_handle = open("Classifier_and_params.pickle","wb")
pickle.dump(dict_to_save, p_handle, protocol=pickle.HIGHEST_PROTOCOL)
p_handle.close()



