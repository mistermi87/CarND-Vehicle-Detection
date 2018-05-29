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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

'''
############ HELPER FUNCTIONS ################
'''
from helper_func import *

'''
############ TRAIN CLASSIFIER ################
'''
#Parameters color only
colorspace_c='RGB'
spatial = 32
histbin = 32
hist_range=(0, 256)

#Parameters HOG
colorspace_h='HLS'
orient = 12
pix_per_cell = 16
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
                        pix_per_cell, cell_per_block, hog_channel)
notcar_features = extract_features(notcars, colorspace_c, (spatial, spatial),
                        histbin, hist_range, colorspace_h, orient, 
                        pix_per_cell, cell_per_block, hog_channel)



if len(car_features) > 0:
    X = np.vstack((car_features, notcar_features)).astype(np.float64) 
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))                   
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else:
    print('Your function only returns empty feature vectors...')
    
    
#evaluate

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler_train = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler_train.transform(X_train)
X_test = X_scaler_train.transform(X_test)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()


svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample

n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])

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
        "svc":svc,\
        "colorspace_c":colorspace_c,\
        "colorspace_h":colorspace_h,}
p_handle = open("Classifier_and_params.pickle","wb")
pickle.dump(dict_to_save, p_handle, protocol=pickle.HIGHEST_PROTOCOL)
p_handle.close()



