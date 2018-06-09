## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/out1.png
[image2]: ./output_images/out2.png
[image3]: ./output_images/out3.png
[image4]: ./output_images/out4.png
[image5]: ./output_images/out5.png
[image6]: ./output_images/out6.png
[image7]: ./output_images/windows.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

For training the classifier(s) in ``train_classifier.py`` I used mostly code given in the lectures. After creating two lists of filenames for training images (vehicle and non-vehicle) using ``glob.glob...`` I did not used the additional data provided in the lecture.
I loaded the images within the function ``extract_features`` in ``helper_func.py``. After optional switching of the colorspace the HOG features are created using
```python
hog_features = get_hog_features(img[:,:,hog_channel], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
```
If the HOG features are taken from all of the three color channels, this function is called in a loop:
```python
if hog_channel == 'ALL':
	hog_features = []
    for channel in range(img.shape[2]):
		hog_features.append(get_hog_features(img[:,:,channel], orient, pix_per_cell, cell_per_block,vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)   
```
I settled for the following parameters of the HOG-feature extraction:

 - colorspace_h='YCrCb'
 - orient = 12
 - pix_per_cell = 8
 - cell_per_block = 2
 - hog_channel = 'ALL'
 
Of course it was a trial and error procedure to get to that values. I noticed that my Laptop began to struggle when creating the ~17500 training examples, if more than ~8000 features where extracted. Since the HOG-features where much more powerful than other features (spatial color, color histogram, described below) they could take the majority of those features. The example in the lecture worked with rather similar values and I found that those were the most reliable ones. However, increasing the Orientation bins to 12 and using the 'YCrCb' colorspace (and 'ALL' of its channel) made the SVM classifier more accurate.
These paraemters create 7056 features per image (7x7x2x2x3x12)

#### 1b. Other features extracted and how they were used
Additionally to the HOG-features I also extracted spatial color features and color histogram features as in the lecture, calling  ``bin_spatial`` and ``color_hist`` from the ``extract_features`` function. For this, the following parameters were used:

- colorspace_c='YCrCb'
- spatial = 16 --> Reduction of Image resolution from 64x64 to 16x16
- histbin = 32 --> 32 bins per histogram (per Color channel)
- hist_range=(0, 256) --> range of the histogram (changing this did not improve the classification)

For experimenting I also created a variable colorspace for the color features. It turned out that it was not really necessary, as it has little to no influence on the result, which colorspace is used.
In order to keep the number of features low the spatial color values are sampled on a reduced resolution of 16x16. The histograms of the distribution in each color channel was set to 32 bins.

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

From ``train_classifier.py`` :
```python
X, y = shuffle(X, y, random_state=0)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

clf1=RandomForestClassifier(n_estimators=10)
clf1.fit(scaled_X[:,0:len_features[0]+len_features[1]-1], y)

clf2 = LinearSVC(C=0.1)
clf2.fit(scaled_X, y)
```

As in the lectures I first fitted a scaler to the training samples. Then I applied the scaler on these samples.

> When finding the best classifier and trying different classifier and feature-extraction parameters, I created a randomized test- and training-set and trained the scaler only on the training features.

Then I introduced two classifier. One classifier is a Linear-Support-vector-machine. Using ``GridsearchCV`` on the C-Parameter, C=1e-5 was suggested. However the best result when training and testing on the final video was between this parameter and the default C=1. Most likely C=1 was slightly overfitting the training data, while C=1e-5 created not enough certainty (decision_function output >>0) for it to be useful in my implementation.
When looking at the different types of features separately it becomes apparent, that the SVM is really good in working with HOG-Features (see also [here](https://arxiv.org/pdf/1406.2419.pdf) ). At the same time it can handle the additional information of the color features just poorly. Because of this, a second classifier is introduced. A random-forrest with n_estimators=10 looks at just the color features (768 spatial + 96 histogram). Combining the Probability of the Result of the two classifiers (~ decision_function of the SVM) is can be seen than the SVM alone has an accuracy of ~99% while a combination of the two classifiers rises the accuracy to ~99.4% which is almost twice as good. Furthermore, it can be later seen that additionally testing just the color features of the sliding window on a 10-layer random-forrest-classifier, comes with almost no speed penalty (when going through the video frames) compared to other measures such as an increase in number of sliding windows.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Before the image (or the first frame of the video) is loaded, the window locations are created by calling following function once:

```python
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
```
with those parameters:
- layers=[48,64,96,128]  --> resolutions
- vert_steps=4           --> vertical steps in each resolution
- y_start=400            --> upper boundary of the area
- y_stop=656             --> lower boundary of the area
- xy_overlap=0.4		--> percentage of overlap between images

This creates a list with ``(x_min,y_min),(x_max,y_max)`` of each window. Here is a demonstration where the images are:
![Windows image][image7]

I try to avoid loops and use arrays or layers (for the different resolutions of the windows) as much as possible.
For this I create a resized image for each resolution of the windows called ``layered_areas_spatial``:
```python
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
```
The function returns a list of zoomed versions of the area of the image in which the windows with each resolution operate.

I did the same with the HOG-Conversions of the image, in order to create ``layered_areas_hog``:
```python
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
```


Having generated this data, the individual computational work done for each window is minimal and the same data can be used for overlapping windows.

```python
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
    #HOG-Area dimensions of the window
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
    
# Define the features vector    
X = np.vstack(window_features).astype(np.float64)
```
Except for the color histogram, all features are now picked from the existing data according to each windows resolution. The color histogram features are created from the original image and scaled such that the same amount of total pixels are distributed between the bins compared to the 64x64px training data

The list of feature vectors is then scaled with the pretrained scaler and used for creating probabilities of detecting a car in each classifier. 
> The SVM does not have a probability function but a ``decision_function`` which is an indicator for the distance of the data point from the multidimensional linear decision surface (--> decision plane). By adding ``+0.5`` and setting all values $<0$ to $0$ a quasi probability function can be achieved with values $>1$ if the calssifier is very certain.
The two probabilites are then merged to ``pt`` by multiplication

Then I introduce a probability threshold and filter only windows/boxes in which there is a high probability that a car is present. The probability vector is filtered out accordingly, leaving only the probable probabilities ``pp``.
```python
scaled_X = X_scaler.transform(X)

p1=svc.decision_function(scaled_X)+0.5 #Measures the probability of each window as the distance from the SVM-Plane
p1[p1<0]=0
p2=rand_forr.predict_proba(scaled_X[:,0:len_features[0]+len_features[1]-1])[:,1]
pt=np.multiply(p1,p2)

prob_thres=1 #probability threshold
bbox_list=[]
for bbox, bit in zip(windows,pt):
    if bit>=prob_thres:
        bbox_list.append(bbox)
pp = pt[(pt >= prob_thres)]
```
Those windows and probabilities are then used to call the function ``heatmap=add_heat2(img, bbox_list,pp)``. (the function is located in  ``helper_func.py``)
In this function an empty heatmap is created and for each of the probable detections the probability is added in the window area. This gives me better results than just adding ``1`` for each detection in the area. For example: One window with a very high certainty of a detected car has now a higher value than two overlaying windows where the classifier is not really certain. 

Finally a bounding  box around each car is created by assigning labels to areas of values>0 using the ``scipy.ndimage.measurements`` ``label``-function. Those labels are then used in the function ``draw_labeled_bboxes`` that I took 1:1 from the lecture.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The aforementioned pipeline with the values described yields to following results:

![First image][image1]
![Second image][image2]
![Third image][image3]
![Fourth image][image4]
![Fifth image][image5]
![Sixth image][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

**Disclaimer**: I tried to use as little parameter tuning as possible to make this pipeline more universally applicable. Because of this, cars on the opposing lanes are detected as well and boxes are displayed. (Due to the smoothing, they seem to appear randomly)
If this video result is not satisfactory, I have stored a classifier and pipeline tuned to this very video with a very smooth output in [a zip](./Version_with_filter.zip) and I am happy to submit a video with an output generated by it.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The workflow for every single frame of the video is identical to the pipeline of a single image. However, in order to smooth the result and suppress occasional false positives I introduced a stack of the last 25 frames (~1s) before the first frame of the video:
```python
img_width=1280
img_height=720
framerate=25
f_smooth=20
heatmap_history=np.zeros((f_smooth,img_height,img_width))
```
The heatmap of a single frame is then stored in this stack in a rolling fashion, with (``t`` is the time of the frame in the video, taken from the video processor):
```python
f=int(t*framerate) #frame number
heatmap_history[f%f_smooth,:,:]=heatmap #using Modulo Operator --> remainder of frame_number/smoothing
```
Then the median is taken for each pixel over the last 20 frames, in order to ignore single false positives. The "median" is not a smooth function compared to the "mean", so sometimes small "islands" of detected cars remain next to the main area of detection in the final result. In order to connect those islands to the corresponding detected car in the final labeling, a Gaussian blur-function with a kernel size of 11 is applied.
```python
heatmap_med=np.median(heatmap_history,axis=0)
heatmap_med=cv2.GaussianBlur(heatmap_med,(11,11),0)
```
From then on, the procedure for the single image is continued ( ``scipy.ndimage.measurements`` ``label``-function & ``draw_labeled_bboxes``)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I used traditional machine learning in this project. Comparing my results to a Convolutional-Neural-Network like [YOLO](https://arxiv.org/abs/1506.02640) it has a lot of drawbacks: 
- In Python it has a speed of ~1.7s/frame (i7-4700mq, 8Gb RAM). A CNN can be up 100x as fast and more (using a parallelization)
- It can only detect cars, not other vehicles or pedestrians

Staying within the realms of traditional machine learning there are still many ways to improve the pipeline:
- When the first car overtakes the second car, the two detections become one big "blob" of car. It would be wise to track the position of each car, because more data could be generated and attributed to it, even while it is indistinguishable from the other car for a few seconds (extrapolated position, behavior attribute... --> "collision course"...)
- The classifier(s) is/are trained on very little data. I could use the given extra data and feed it into the training pipeline, but it would take too much additional time
- Also I could have picked some of the false positives and trained the classifier to avoid classifying those as cars.
- The sliding window operation could be modified depending on the localization of positive detections in previous frames (--> "looking at it more thoroughly")

Issues I faced:
Speaking about the parameter tuning.... For me, the video had two problematic timeframes:
1. A lot of median highway boundary was detected after the first car entered the scene.
2. Before the second car enters the scene the white car is quite far away. It is not detected very well then for a few images
At some point I felt I could not improve the classifiers any more and I had to find the right balance between setting a low or high probability threshold (--> better car detection vs. better false positive mitigation).
I now tuned the parameter for this very video, which was a tedious and time consuming matter. Also deciding on the best parameters of the sliding windows locations (speed vs. accuracy) took some time.
In a first version I had even more parameters (e.g. deleting all values $>0.5$ in the heatmap before labeling). While this  version gave a cleaner output it was very much tuned to this very video, which was personally not satisfactory.
