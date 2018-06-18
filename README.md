# Dynamic Gesture Recognition using Leap Motion

Based on the work done [here](http://scholarworks.rit.edu/cgi/viewcontent.cgi?article=1877&context=other) it is proposed to use trajectory based images to discriminate between dynamic gestures found at this [Dataset](https://github.com/rcmccartney/DataCollector).

Here it is used only the Finger Tip Positions (and the Palm position) to get spatial and temporal information of a given gesture. It is generated an Image by each Orthogonal Plane in the space (XY, XZ and ZY planes). This images are used as input to feed a CNN.

**Recognizer directory**: contains the model used to classify the data as well as a Python Script to perform online recognition (this Script requires TensorFlow to work properly).

**Visualize directory**: contains a Processing sketch with a pretty simple GUI that allows to identify Dynamic Gestures done by the user. To this purpose gesture detection (segmentation) is done when the user keeps her hand visible and makes no movement in a short period of time.

Processing sketch (Client) and Python program (Server) communicate by sockets. Whenever a gesture is done, this information is send to the Server and the Client will receive which is the most likely gesture done by the user.
