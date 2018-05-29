# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:45:15 2018

@author: mstei
"""

import numpy as np
import cv2
import matplotlib.image as mpimg
#import pickle
from skimage.feature import hog


def convert_color(img, conv):
    if conv != 'RGB':
        if conv == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif conv == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif conv == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif conv == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif conv == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: return img


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace_c, spatial_size, hist_bins, hist_range, cspace_h, orient, 
                        pix_per_cell, cell_per_block, hog_channel):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img_f in imgs:
            
    # Read in each one by one
        image=cv2.imread(img_f)
        # apply color conversion if other than 'RGB'
        if cspace_c != 'RGB':
            if cspace_c == 'HSV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace_c == 'LUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace_c == 'HLS':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace_c == 'YUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace_c == 'YCrCb':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: img = np.copy(image)
        ######Color features          
        # Apply bin_spatial() to get spatial color features
        s_features=bin_spatial(img, size=spatial_size)
        # Apply color_hist() to get color histogram features
        c_features=color_hist(img, nbins=hist_bins, bins_range=hist_range)
        
        if cspace_h != 'RGB':
            if cspace_h == 'HSV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace_h == 'LUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace_h == 'HLS':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace_h == 'YUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace_h == 'YCrCb':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: img = np.copy(image)          
        ######HOG features
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
        # Append the new feature vector to the features list
        feature=np.concatenate((s_features,c_features, hog_features))
        
        features.append(feature)
        print(img_f)
    # Return list of feature vectors
    return features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        #color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
        #thick=np.random.randint(1,7)
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# =============================================================================
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
#                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0]==None: x_start_stop[0]=0
#     if x_start_stop[1]==None: x_start_stop[1]=img.shape[1]
#     if y_start_stop[0]==None: y_start_stop[0]=0
#     if y_start_stop[1]==None: y_start_stop[1]=img.shape[0]
#     # Compute the span of the region to be searched
#     span_x=x_start_stop[1]-x_start_stop[0]
#     span_y=y_start_stop[1]-y_start_stop[0]
#     # Compute the number of pixels per step in x/y
#     pps_xy=np.multiply(xy_window,xy_overlap)
#     # Compute the number of windows in x/y
#     windows_x = 1 + (span_x - xy_window[0])/(xy_window[0] * xy_overlap[0])
#     windows_y = 1 + (span_y - xy_window[1])/(xy_window[1] * xy_overlap[1])
#     i_windows_x=np.floor(windows_x).astype(int)
#     i_windows_y=np.floor(windows_y).astype(int)
#     r_windows_x=windows_x-i_windows_x
#     r_windows_y=windows_y-i_windows_y
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     for i in range(i_windows_x):
#         for j in range(i_windows_y):
#             position_x=int(x_start_stop[0]+i*xy_window[0]*xy_overlap[0])
#             position_y=int(y_start_stop[0]+j*xy_window[1]*xy_overlap[1])
#             window_list.append(((position_x,position_y),(position_x+xy_window[0],position_y+xy_window[1])))
#         if r_windows_y>0:
#             position_x=int(x_start_stop[0]+i*xy_window[0]*xy_overlap[0])
#             position_y=int(y_start_stop[1]-xy_window[1])
#             window_list.append(((position_x,position_y),(position_x+xy_window[0],position_y+xy_window[1])))
#     if r_windows_x>0:
#         for j in range(i_windows_y):
#             position_x=int(x_start_stop[1]-xy_window[0])
#             position_y=int(y_start_stop[0]+j*xy_window[1]*xy_overlap[1])
#             window_list.append(((position_x,position_y),(position_x+xy_window[0],position_y+xy_window[1])))
#     
#             
#         # Note: you could vectorize this step, but in practice
#         # you'll be considering windows one by one with your
#         # classifier, so looping makes sense
#         # Calculate each window position
#         # Append window position to list
#     # Return the list of windows
#     return window_list
# =============================================================================

def slide_window2(img, layers,vert_steps,y_start,y_stop, xy_overlap):
    window_list=[]
    # If x and/or y start/stop positions not defined, set to image size
    for xy_window in layers:
        for i in range(vert_steps):
            position_x1=np.arange(0,img.shape[1]-int(xy_window*xy_overlap),int(xy_window*xy_overlap))
            position_x2=position_x1+xy_window
            position_y1=(y_start*np.ones_like(position_x1)+i*xy_window*xy_overlap).astype(int)
            position_y2=position_y1+xy_window
            if position_y2[0]<=y_stop:
                window_list.extend(list(zip(list(zip(position_x1,position_y1)),list(zip(position_x2,position_y2)))))
    return window_list

def get_spatial_whole_area(image, cspace_c,spatial_size, y_start, y_stop, layers, xy_overlap, vert_steps):
    if cspace_c != 'RGB':
            if cspace_c == 'HSV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace_c == 'LUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace_c == 'HLS':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace_c == 'YUV':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace_c == 'YCrCb':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: img = np.copy(image)
    # resize image
    whole_area_spatial=[]
    for layer in layers:
        resize_factor=spatial_size/layer
        y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
        layer_spatial=cv2.resize(img[y_start:y_stop_n,:,:],None,fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)
        whole_area_spatial.append(layer_spatial)
    return whole_area_spatial

def get_hog_whole_area(image, orient, pix_per_cell, cell_per_block, cspace_h, hog_channel, y_start, y_stop, layers, xy_overlap, vert_steps):
    #change color of frame for hog_analysis, according to training features
    if cspace_h != 'RGB':
        if cspace_h == 'HSV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace_h == 'LUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace_h == 'HLS':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace_h == 'YUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace_h == 'YCrCb':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: img = np.copy(image)
        
    # Create HOG for area
    whole_area_hog=[] 
    if hog_channel == 'ALL':
         for layer in layers:
             y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
             pix_per_cell_n=int(pix_per_cell*layer/64) #64=size of training data=same size of sliding windows
             print(pix_per_cell_n, y_stop_n)
             layer_hog = []
             for channel in range(img.shape[2]):
                 layer_hog.append(hog(img[y_start:y_stop_n,:,channel], orientations=orient, 
                                      pixels_per_cell=(pix_per_cell_n, pix_per_cell_n),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm= 'L2-Hys', transform_sqrt=False, 
                                      visualise=False, feature_vector=False))
             whole_area_hog.append(layer_hog)    
    else:
         for layer in layers:
             y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
             pix_per_cell_n=int(pix_per_cell*layer/64)
             layer_hog = hog(img[y_start:y_stop_n,:,hog_channel], orientations=orient, 
                             pixels_per_cell=(pix_per_cell_n, pix_per_cell_n),
                             cells_per_block=(cell_per_block, cell_per_block),
                             block_norm= 'L2-Hys', transform_sqrt=False, 
                             visualise=False, feature_vector=False)
             whole_area_hog.append(layer_hog)
    return whole_area_hog
    