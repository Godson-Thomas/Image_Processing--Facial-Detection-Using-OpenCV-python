<img src="https://github.com/Godson-Thomas/Data-transfer-between-Arduino-UNO-and-ESP8266/blob/master/UNO_to_ESP8266/Images/P.png" width="400"> <br><br>
# COMPUTER VISION
Computer Vision, often abbreviated as CV, is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.Moreover Computer vision focuses on replicating parts of the complexity of the human vision system and enabling computers to identify and process objects in images and videos in the same way that humans do.<br>
Here we are going to detect face from an image by using **HAAR CASCADE** classifier.
# HAAR CASCADE
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.<br>
OpenCV already contains many pre-trained classifiers for face, eyes, smile etc..So we will be using one of these.
# Library
OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision. In short it is a library used for Image Processing. It is mainly used to do all the operation related to Images.We will be using this library.
# Steps
## Installation


1. We will be using Jupyter Notebook for writing the code.Make sure you have Jupyter Notebook installed.<br><br>
2. Lauch your Jupyter Notebook<br><br>
3. Now we have to install the OpenCV library.Type the code in the cell of Jupyter Notebook and run it.
```
pip install opencv-python
```
<br>
<img src="https://github.com/Godson-Thomas/Data-transfer-between-Arduino-UNO-and-ESP8266/blob/master/UNO_to_ESP8266/Images/P.png" width="400">  <br><br> 

4. Download the HAAR CASCADE. [click here]()<br><br>
* ## Code
5. Import the OpenCV library
```
import cv2
```
6. Load your image which is to be detected to a variable using this code.
```
img=cv2.imread("/CV-ImageF/Images/Tesla.jpg") # Image location
```
7. Now read the Haar Cascade classifier.
```
face_cascade=cv2.CascadeClassifier("/CV-ImageF/_.xml/haarcascade_frontalface_default.xml")   # Classifier Location
```
8. You can resize the image if you want.
```
r=500/img.shape[1]                          #RESIZING THE LOADED IMAGE
print(r)
dim=(500,int(img.shape[0]*r))
```
9. Storing and displaying the resized image.
```
resized_img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)  
cv2.imshow("RESIZED ONE",resized_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
```
Here we have used a **cv2.waitkey(0)** to exit from the displayed tab on pressing any key.<br><br>
10. Now we have to convert the image into a Gray scale image for the algorithm to perform efficiently.
```
gray_img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
```
 11. Now we perform our algorithm on it.
 ```
 faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.05 ,minNeighbors=6)
 ```
 12. We have to draw a rectangle in the region where face is detected.You can draw any shape you want.
 ```
 for x,y,w,h in faces:
    
    final_img=cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3)

cv2.imshow("DETECTED FRAME",final_img)

cv2.waitKey(0)

cv2.destroyAllWindows()
```
<br>
<img src="https://github.com/Godson-Thomas/Data-transfer-between-Arduino-UNO-and-ESP8266/blob/master/UNO_to_ESP8266/Images/P.png" width="400">


13. You can also display the rectangle in the coloured image by replacing 
```
 final_img=cv2.rectangle(resized_img,(x,y),(x+w,y+h),(255,0,0),3)
 ```
 ### Note : 
 Coloured images are 3D numpy array while the other is 2D. So when there are lots of images, it is better to convert it to a gray scale image for the algorithm to perform calculations effectively.<br><br>  <img src="https://github.com/Godson-Thomas/Data-transfer-between-Arduino-UNO-and-ESP8266/blob/master/UNO_to_ESP8266/Images/P.png" width="400">  <br><br>
