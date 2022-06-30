
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import time
import cv2 
import os


Knife_detector = load_model("knife_detector.model5")

vs = VideoStream(src=0).start()

while True:
    
    frame = vs.read()
    
    image=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
  
    image = img_to_array(image)  
    image = preprocess_input(image)

    image = image.reshape(1,224,224,3)
    predictions = Knife_detector.predict(image, batch_size=32)

    predictions = np.argmax(predictions, axis=1)
    if(predictions==1):
        print('Safe:No weapon Detected')
    else:
        print('DANGER!! KNIFE DETECTED')

    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()