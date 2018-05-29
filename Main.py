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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import glob

'''
############ HELPER FUNCTIONS ################
'''
from helper_func import *


'''
############ Image Pipeline ################
'''

# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load( open("Classifier_and_params.pickle", "rb" ) )

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial"]
hist_bins = dist_pickle["histbin"]
hog_channel = dist_pickle["hog_channel"]
#colorspace_c=dist_pickle["colorspace_c"]
#colorspace_h=dist_pickle["colorspace_h"]

img = mpimg.imread('test_images/test1.jpg')

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, colorspace_c, colorspace_h, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=colorspace_c)
    htrans_tosearch = convert_color(img_tosearch, conv=colorspace_h)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, None, fx=int(1/scale), fy=int(1/scale), interpolation = cv2.INTER_AREA)
        htrans_tosearch = cv2.resize(htrans_tosearch, None, fx=int(1/scale), fy=int(1/scale), interpolation = cv2.INTER_AREA)
        
    ch1 = htrans_tosearch[:,:,0]
    ch2 = htrans_tosearch[:,:,1]
    ch3 = htrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            print(subimg.shape)
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img


#windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 650], xy_window=(32, 32), xy_overlap=(1, 1))
layers=[32,64,128,256]
vert_steps=4
y_start=400
y_stop=656
xy_overlap=0.5

####
# =============================================================================
# img_w_cars=[]
# for layer in layers:
#     scale=layer/64
#     y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
#     image=find_cars(img, y_start, y_stop_n, scale, colorspace_c, colorspace_h, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# =============================================================================



windows = slide_window2(img, layers,vert_steps, y_start,y_stop, xy_overlap)
print(len(windows))
                      
window_img = draw_boxes(img, windows, color=(0, 0, 255), thick=1)                    
plt.imsave('foo.png',window_img)

#precompute resized image for spatial color features and hog
layered_areas_spatial, layered_areas_chist= get_spatial_whole_area(img, colorspace_c, spatial_size, y_start, y_stop, layers, xy_overlap, vert_steps) 
layered_areas_hog= get_hog_whole_area(img, orient, pix_per_cell, cell_per_block, colorspace_h, hog_channel, y_start, y_stop, layers, xy_overlap, vert_steps)


window_features=[]
for window in windows:
    window_size=window[1][0]-window[0][0] #x2-x1
    layer_num=np.searchsorted(layers, window_size)
    x1=window[0][0]
    y1=window[0][1]-y_start
    x2=window[1][0]
    y2=window[1][1]-y_start
    
    pix_per_cell_n=int(pix_per_cell*(layers[layer_num]/64))
    hoga_x2=int(x2/pix_per_cell_n-cell_per_block)
    hoga_x1=int(hoga_x2-(window_size/pix_per_cell_n-cell_per_block))
    hoga_y2=int(y2/pix_per_cell_n-cell_per_block)
    hoga_y1=int(hoga_y2-(window_size/pix_per_cell_n-cell_per_block))
    
    if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(layered_areas_hog[layer_num][channel][hoga_y1:hoga_y2,hoga_x1:hoga_x2,:,:,:])
            hog_features = np.ravel(hog_features)        
    else:
        hog_features = layered_areas_hog[layer_num][hoga_y1:hoga_y2,hoga_x1:hoga_x2,:,:,:]
        hog_features = np.ravel(hog_features)
    
    zoom=window_size/spatial_size
    x1s=int(x1*zoom)
    x2s=int(x2*zoom)
    y1s=int(y1*zoom)
    y2s=int(y2*zoom)
    spatial_features=np.hstack((layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,0].ravel(),\
                               layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,1].ravel(),\
                               layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,2].ravel()))
    
    window_features.append(np.concatenate(spatial_features,hog_features))
    #to do schauen nach dateiformat in training 0-1 0-255....
#           
##out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#
##plt.imshow(out_img)