# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:45:15 2018

@author: mstei
"""

import numpy as np
import cv2
from skimage.feature import hog


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
        c_features=color_hist(image, nbins=hist_bins, bins_range=hist_range)
        # apply color conversion if other than 'BGR'
        
        if cspace_c != 'BGR':
            if cspace_c == 'YCrCb':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif cspace_c == 'LUV':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace_c == 'HLS':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace_c == 'YUV':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace_c == 'RGB':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else: img = np.copy(image)
        ######Color features          
        # Apply bin_spatial() to get spatial color features
        s_features=bin_spatial(img, size=spatial_size)
        # Apply color_hist() to get color histogram features
        
        
        if cspace_h != 'BGR':
            if cspace_h == 'YCrCb':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif cspace_h == 'LUV':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace_h == 'HLS':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace_h == 'YUV':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace_h == 'RGB':
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        #print(len(hog_features))
        feature=np.concatenate((s_features,c_features, hog_features))
        
        features.append(feature)
        print(img_f)
    # Return list of feature vectors
    return features, (len(s_features),len(c_features),len(hog_features))

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


def slide_window(img_width, layers,vert_steps,y_start,y_stop, xy_overlap):
    window_list=[]
    # If x and/or y start/stop positions not defined, set to image size
    for xy_window in layers:
        for i in range(vert_steps):
            position_x1=np.arange(0,img_width-int(xy_window)+1,int(xy_window*(xy_overlap)))
            position_x2=position_x1+xy_window
            position_y1=(y_start*np.ones_like(position_x1)+i*xy_window*(xy_overlap)).astype(int)
            position_y2=position_y1+xy_window
            if position_y2[0]<=y_stop:
                window_list.extend(list(zip(list(zip(position_x1,position_y1)),list(zip(position_x2,position_y2)))))
    return window_list

def get_spatial_whole_area(image, cspace_c,spatial_size, y_start, y_stop, layers, xy_overlap, vert_steps):
    if cspace_c != 'RGB':
        if cspace_c == 'YCrCb':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace_c == 'LUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace_c == 'HLS':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace_c == 'YUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace_c == 'BGR':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else: img = np.copy(image)
    # resize image
    whole_area_spatial=[]
    for layer in layers:
        resize_factor_s=spatial_size/layer
        #resize_factor_ch=64/layer
        y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
        layer_spatial=cv2.resize(img[y_start:y_stop_n,:,:],None,fx=resize_factor_s, fy=resize_factor_s, interpolation = cv2.INTER_AREA)
        whole_area_spatial.append(layer_spatial)
    return whole_area_spatial



def get_hog_whole_area(image, orient, pix_per_cell, cell_per_block, cspace_h, hog_channel, y_start, y_stop, layers, xy_overlap, vert_steps):
    #change color of frame for hog_analysis, according to training features
    if cspace_h != 'RGB':
        if cspace_h == 'YCrCb':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace_h == 'LUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace_h == 'HLS':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace_h == 'YUV':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace_h == 'BGR':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
    else: img = np.copy(image)
        
    # Create HOG for area with each layer zoomed acording to training size
    whole_area_hog=[] 
    if hog_channel == 'ALL':
         for layer in layers:
             y_stop_n=np.min([int(y_start+layer*xy_overlap*(vert_steps-1)+layer),y_stop])
             pix_per_cell_n=int(pix_per_cell*layer/64) #64=size of training data=same size of sliding windows
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
 
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def add_heat2(img, bbox_list,p):
    #create empy heatmap
    heatmap=np.zeros_like(img[:,:,0]).astype(np.float64)
    # Iterate through list of bboxes
    for prob, box in zip(p,bbox_list):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += prob

    # Return updated heatmap
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img