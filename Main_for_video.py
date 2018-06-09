# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:32:44 2018

@author: mstei
"""

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
#from sklearn.preprocessing import StandardScaler
#import glob
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

'''
############ HELPER FUNCTIONS ################
'''
from helper_func import *



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

#Sliding windows parameters
layers=[48,64,96,128]  #resolutions
vert_steps=4           #vertical steps in each resolution
y_start=400            #upper boundary of the area
y_stop=656             #lower boundary of the area
xy_overlap=0.4

img_width=1280#img.shape[1]
img_height=720#img.shape[0]
framerate=25

windows = slide_window(img_width, layers, vert_steps, y_start,y_stop, xy_overlap)



f_smooth=20
heatmap_history=np.zeros((f_smooth,img_height,img_width))




'''
#######Things that are done for every frame########
'''
def process_frame(get_frame,t):
    
    f=int(t*framerate) # frame number
    
    #precompute resized image for spatial color features and hog
    img=get_frame(t)
    
    layered_areas_spatial= get_spatial_whole_area(img, colorspace_c, spatial_size, y_start, y_stop, layers, xy_overlap, vert_steps) 
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
                    hog_features.append(layered_areas_hog[layer_num][channel][hoga_y1:hoga_y2+1,hoga_x1:hoga_x2+1,:,:,:])
                hog_features = np.ravel(hog_features)        
        else:
            hog_features = layered_areas_hog[layer_num][hoga_y1:hoga_y2,hoga_x1:hoga_x2,:,:,:]
            hog_features = np.ravel(hog_features)
        
        zoom=spatial_size/window_size
        x1s=int(x1*zoom)
        x2s=int(x2*zoom)
        y1s=int(y1*zoom)
        y2s=int(y2*zoom)
        spatial_features=np.hstack((layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,0].ravel(),\
                                   layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,1].ravel(),\
                                   layered_areas_spatial[layer_num][y1s:y2s,x1s:x2s,2].ravel()))
        chist_features=(color_hist(img[window[0][1]:window[1][1],window[0][0]:window[1][0],:], hist_bins, hist_range))/(window_size**2)*(64**2) #norm the histogram as if the window had 64x64 px
        
        window_features.append(np.concatenate((spatial_features,chist_features,hog_features)))
    
    # Define the features vector    
    X = np.vstack(window_features).astype(np.float64)
    # Define the labels vector
    #y = np.ones(len(window_features))
    
    
    scaled_X = X_scaler.transform(X)
    p1=svc.decision_function(scaled_X)+0.5
    p1[p1<0]=0

    p2=rand_forr.predict_proba(scaled_X[:,0:len_features[0]+len_features[1]-1])[:,1]

    pt=np.multiply(p1,p2)

    prob_thres=1 #probability threshold
    bbox_list=[]
    for bbox, bit in zip(windows,pt):
        if bit>=prob_thres:
            bbox_list.append(bbox)
    pp = pt[(pt >= prob_thres)]

    heatmap=add_heat2(img, bbox_list,pp)
     
    heatmap_history[f%f_smooth,:,:]=heatmap
    
    heatmap_med=np.median(heatmap_history,axis=0)#####mean
    
    #heatmap_med[(heatmap_med < 1)]=0
    #heatmap_med[(heatmap_med > 5)]=5
    heatmap_med=cv2.GaussianBlur(heatmap_med,(11,11),0)
    
    #heatmap[(heatmap > 2)]=2

    #
    labels = label(heatmap_med)
    draw_img = draw_labeled_bboxes(np.copy(get_frame(t)), labels)
    
    #draw_img=draw_boxes(get_frame(t), bbox_list, color=(0, 0, 255), thick=6)
    return draw_img

video_output = 'output_images/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl(process_frame)
white_clip.write_videofile(video_output, audio=False)
