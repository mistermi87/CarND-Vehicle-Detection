# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:01:44 2018

@author: mstei
"""

"""
#Pre Workflow for filters
Train color features
Train HOG-features
Normalize features

#Workflow in each frame
1. Create Criteria from color filter
2. Create Criteria from HOG features
3. Create vector with each feature concatenated
Normalize vectors
Sliding windows with vector each
Linear SVM to compare vectors
Deal with Multiple Detections and False Positives --> Heat Map
"""

import numpy as np
import cv2
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
#from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
#import glob
import time
import seaborn as sns




'''
############ HELPER FUNCTIONS ################
'''
from helper_func import *


'''
############ Image Pipeline ################
'''

'''
#######Things that are done once, globally########
'''

# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load( open("Classifier_and_params.pickle", "rb" ) )

# get attributes of our svc object
svc = dist_pickle["svc"]
rand_forr = dist_pickle["rand_forr"]
len_features = dist_pickle["len_features"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial"]
hist_bins = dist_pickle["histbin"]
hist_range = dist_pickle["hist_range"]
hog_channel = dist_pickle["hog_channel"]
colorspace_c=dist_pickle["colorspace_c"]
colorspace_h=dist_pickle["colorspace_h"]

img = cv2.imread('test_images/test6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

layers=[48,64,96,128]
vert_steps=4
y_start=400
y_stop=656
xy_overlap=0.4

img_width=img.shape[1]
img_height=img.shape[0]

windows = slide_window(img_width, layers, vert_steps, y_start,y_stop, xy_overlap)

                      
#window_img = draw_boxes(img, windows, color=(0, 0, 255), thick=1)                    
#plt.imsave('foo.png',window_img)
start_time = time.time()
'''
#######Things that are done for every frame########
'''

#precompute resized image for spatial color features and hog
layered_areas_spatial= get_spatial_whole_area(img, colorspace_c, spatial_size, y_start, y_stop, layers, xy_overlap, vert_steps) 
print("--- %s seconds ---" % (time.time() - start_time))
layered_areas_hog= get_hog_whole_area(img, orient, pix_per_cell, cell_per_block, colorspace_h, hog_channel, y_start, y_stop, layers, xy_overlap, vert_steps)
print("--- %s seconds ---" % (time.time() - start_time))

window_features=[]
for window in windows:
    window_size=window[1][0]-window[0][0] #x2-x1
    layer_num=np.searchsorted(layers, window_size) #what layer to call
    #borders of the windows 
    x1=window[0][0]
    y1=window[0][1]-y_start
    x2=window[1][0]
    y2=window[1][1]-y_start
    #####   HOG   #####
    pix_per_cell_n=int(pix_per_cell*(layers[layer_num]/64)) #"zoom" of the HOG features
    hoga_x2=int(x2/pix_per_cell_n-cell_per_block)
    hoga_x1=int(hoga_x2-(window_size/pix_per_cell_n-cell_per_block))
    hoga_y2=int(y2/pix_per_cell_n-cell_per_block)
    hoga_y1=int(hoga_y2-(window_size/pix_per_cell_n-cell_per_block))
    
    if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(layered_areas_hog[layer_num][channel][hoga_y1:hoga_y2+1,hoga_x1:hoga_x2+1,:,:,:])
            hog_features = np.ravel(hog_features)        
    else:
        hog_features = layered_areas_hog[layer_num][hoga_y1:hoga_y2,hoga_x1:hoga_x2,:,:,:]
        hog_features = np.ravel(hog_features)
    #####   Spatial Color   #####
    zoom=spatial_size/window_size
    x1s=int(x1*zoom)
    x2s=int(x2*zoom)
    y1s=int(y1*zoom)
    y2s=int(y2*zoom)
    spatial_features=np.hstack((layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,0].ravel(),\
                               layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,1].ravel(),\
                               layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,2].ravel()))
    #####   Color histogram   #####
    chist_features=(color_hist(img[window[0][1]:window[1][1],window[0][0]:window[1][0],:], hist_bins, hist_range))/(window_size**2)*(64**2) #norm the histogram as if the window had 64x64 px
    
    window_features.append(np.concatenate((spatial_features,chist_features,hog_features)))
print("--- %s seconds ---" % (time.time() - start_time))


# Define the features vector    
X = np.vstack(window_features).astype(np.float64)
# Define the labels vector
y = np.ones(len(window_features))

scaled_X = X_scaler.transform(X)

p1=svc.decision_function(scaled_X)+0.5
p1[p1<0]=0
p2=rand_forr.predict_proba(scaled_X[:,0:len_features[0]+len_features[1]-1])[:,1]

pt=np.multiply(p1,p2)

print("--- %s seconds ---" % (time.time() - start_time))
prob_thres=1 #probability threshold
bbox_list=[]
for bbox, bit in zip(windows,pt):
    if bit>=prob_thres:
        bbox_list.append(bbox)
pp = pt[(pt >= prob_thres)]
    
out=draw_boxes(img, bbox_list, (0, 0, 255), 6)
out=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
plt.imsave('foo2.png',out)
print("--- %s seconds ---" % (time.time() - start_time))

#heatmap=np.zeros_like(img[:,:,0]).astype(float)
heatmap=add_heat2(img, bbox_list,pp)
#heatmap=add_heat(heatmap, bbox_list)
#heatmap[(heatmap < 2)]=0
#heatmap[(heatmap > 10)]=10
labels2 = label(heatmap)
draw_img1 = draw_boxes(np.copy(img), bbox_list, color=(0, 0, 255), thick=6)
draw_img2 = draw_labeled_bboxes(np.copy(img), labels2)
#heatmap=cv2.GaussianBlur(heatmap,(101,101),0)


print("--- %s seconds ---" % (time.time() - start_time))

fig = plt.figure(figsize=(20,6))
plt.subplot(131)
plt.imshow(draw_img1)
plt.title('All positive windows')
plt.subplot(132)
sns.heatmap(heatmap, cbar = True, square = True, annot=False,yticklabels=False,xticklabels=False,  cmap= 'inferno')
#plt.show()
plt.title('Filtered heatmap')
plt.subplot(133)
plt.imshow(draw_img2)
plt.title('The car(s) detected')
plt.tight_layout()

fig.savefig('plot.png')






#           
##out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#
##plt.imshow(out_img)