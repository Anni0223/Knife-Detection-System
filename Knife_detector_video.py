# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2 
import os



# load the face mask detector model from disk
Knife_detector = load_model("knife_detector.model3")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    image=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
    
   # (h, w) = frame.shape[:2]
    #blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
       # (104.0, 177.0, 123.0))
    image = img_to_array(image)   # Converting image to an array.
    image = preprocess_input(image)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
   # preds = Knife_detector.predict(image)
    image = image.reshape(1,224,224,3)
    predictions = Knife_detector.predict(image, batch_size=32)

    predictions = np.argmax(predictions, axis=1)
    if(predictions==1):
        print('Safe:No weapon Detected')
    else:
        print('DANGER!! KNIFE DETECTED')

    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()