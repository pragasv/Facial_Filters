# Facial-filters


Over here we try to fit face filters ,very similar to what we normally see in 
snapchat or any other apps. I have worked on a very specific type of filter, which is 
eye wear.

eg - normal glasses , sunglasses etc

Looking at the usage, the inital idea is to implement this live using a live webcam feed. 
uncomment the line to run the code with the usage of webcam. 

I have used the dlib features than the cascaded haar filters, as it was more accurate and fast. 
We have 63 face feature points, the filter is added using some of these points. 

The code isnt supportive to faces in the wild. The image used / the should be sufficiently focused towards the webcam. 

Additionally the output also has the eyes , eyebrows and mouth marked. This is just for a reference for other types of filters.
Uncomment the lines if u dont want this in the final product. 