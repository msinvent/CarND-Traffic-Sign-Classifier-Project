## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./germanTrafficSign/50SpeedLimit.jpg "SpeedLimit of 50"
[image2]: ./germanTrafficSign/noentry.jpg "No entry"
[image3]: ./germanTrafficSign/stop.jpg "Stop"
[image4]: ./germanTrafficSign/straightOnly.jpg "Ahead Only"
[image5]: ./germanTrafficSign/workOnRoad.jpg "Road Work"
[image6]: ./germanTrafficSign/yield.jpg "Yield"

Project Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

## The Project
---
The goals / steps of this project are the following:
### Load the data set
---
I have used pickle library to read the data from the already preprocessed data given to us.


### Explore, summarize and visualize the data set
---
Followed by which I have simply looked at the data size of the training and testing data set along with the image size to resructure the deep network accordingly. Each image proved out to be 32x32x3, thus I first resized the images to a gray scale image to attemp to use Lenet architecture initially and to then slowly build from there.

We can see a lot of variation in the frequency of training samples of respective classes with frequency of different classes given as :  
``` 
[ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920
  690  540  360  990 1080  180  300  270  330  450  240 1350  540  210
  480  240  390  690  210  599  360 1080  330  180 1860  270  300  210
  210 ]
```

where class 0 'Speed limit (20km/h)' with 180 samples, class 1 'Speed limit (30km/h)' with 1980 samples and so on and so forth. Though there is significant difference in the frequency I am not going for data augmentation of balancing because of my assumptionn that if the training and test data set is part of a single sample pool. And if the process of selection is random to a good extent then any bias introduced in the network due to biased class sample frequency will not have a significant affect since even the training data will have a similar frequency distribution. ( It is just an assumptions and I might would have come back to it if the accurary requirement after all my other effors would not have would not have given me satisfactorily result )


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

### Design, train and test a model architecture
As I mentioned above the first step of data preprocessing was to change the image to gray scale using 
```
X = 0.2989*X[:,:,:,0] + 0.5870*X[:,:,:,1] + 0.1140*X[:,:,:,2]
```
Ref : https://www.mathworks.com/matlabcentral/answers/196535-function-to-convert-rgb-to-grayscale

Then I trained the network on the Lenet to se the training and validation accurary, which was around 89%, as mentioned in the lecture notes. To gain more accuracy the first thing that I thought of is that in comparison to LeNet now we have much more classes to clasify 43 in comparison to 10, which pointed to the requirement of more parameters. To make sure I am not overtraining over my data set since I was adding bunch of extra parameters I added dropout layers after each layer. It took me some time to reach the current network configuration with a lot of hit and trial involved.
A couple of observations that I want to mention is that dropout of 0.50 was not able to even give good accuracy on the training data, that points to number of parameters required to be more.

The exact network graph is :


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, output 28x28x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 14x14x18 				|
|     |       |
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x18 	|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
| Max pooling	      	| 2x2 stride,  output 5x5x18 				|
|     |       |
| Flatten             | output 450x1  |
| Fully connected		| output 240x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
|     |       |
| Fully connected		| output 129x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|
|     |       |
| Fully connected		| output 43x1 							|
| RELU					|												|
| Dropout	      	| Keep Probabiligy = 0.80 				|

Accuracy on validation dataset around 93 - 96%
Accuracy on trainig dataset around 94%
Accuracy on internet scrapped images : 4/6

### Use the model to make predictions on new images
I scraped the internet to find the images with google image search using the keyword 'german traffic signs' it turns out this was a task in itself since most of these searches were taking me to the same dataset. I tried my best to make sure that the images I am getting are not part of the training or validation set.

Just to give an idea about how varied these images are I am plotting them in their original proportion
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

My second task was to reshape the images to use in my trained network. I tried using 
```
image = cv2.resize(image, (32, 32)) 
```
On the image after transforming it the grayscale but got bad results in terms of prediction, probably because of very different form factors, thus I choose to manually crop the images to square shape by careful selection of cropping area. I believe this will not be a part of a deep neural network design of training cycle as in that case I will suppose most of my data set is coming from a very similar camera sources.

After mamnual data shaping I fed the images inside the trained neural network to get 4 out of 6 images predicted correctly. The predicted classes are

For the class  straightOnly  the top classes with corresponding softmax probabilities 


| Class Name         		|     softMax Probability	        					|   logit value  |
|:---------------------:|:---------------------------------------------:|:--------|
| Ahead only |   1.0  |  0.06681505 |
| Yield |   0.0  | 0.04762857 |
| Go straight or right |   0.0  | 0.04726604 |
| Turn right ahead |   0.0  | 0.04079577 |
| Priority road |   0.0  | 0.03750502 |



### Dataset and Repository

2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Possible Improvements
We can definitely get better results if we find a way to augment the dataset with more data. That can be done for some of the signs by simply taking a mirror image(but be careful with the new labels)
Some other affine or projective transforms can also be applied to create new samples. 

