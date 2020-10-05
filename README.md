## Smile Detector v1a.

#### This is a realtime AI smile detecting application written on python using the openCV library.

###### Program is based on the Viola-Jones algorithm.

###### It detects faces and upon successful face detection looks for a smile inside those faces. When the AI finds a smile inside the face, it then outputs a message under the smile.

###### We have used a number of optimization techniques such as making the color feed / image into black and white so it detects faces easier. We have also set restrictions on smile finding based upon where a smile can occur. This application is built upon the thesis that a smile can only appear on a face (I have ignored billboards for teeth whitening and dental advertisements).



## Example

![Obama Happy](https://i.imgur.com/bvWp8Ow.png)

###### Program is well documented. Works in realtime using the camera of your device but it also operates on images and videos.
