import numpy as np
import mediapipe as mp 
import cv2
from PIL import Image

def detect_faces(image):#image - PIL Image object
    image_np=np.array(image)#image_np- numpy array representation of the image(pil)
    face= None

    mp_face_detection=mp.solutions.face_detection
    #soluions is a collection of pre-trained AI models.( hands ,pose,facemesh,face detection)
    
    face_detector=mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5)
#as we can see in FaceDetection F is capital ie its a class

    results = face_detector.process(image_np) 
    #process- processes the image and returns the detection results
    #result-Information about detected faces(cordinates ,confidence)
    if results.detections:
      for detection in results.detections:
        bbox=detection.location_data.relative_bounding_box
        h, w, _=image_np.shape
        x=int(bbox.xmin * w)
        y=int(bbox.ymin * h)
        width=int(bbox.width * w)
        height=int(bbox.height * h)

        #bbox-bounding box-MediaPipe doesn't return the face image.
        # Instead, it returns a rectangle around the face.
        # inside bbox-facelocation info(xmin,ymin,width,height)
        
        cv2.rectangle(
            image_np,# to draw rectangle on numpy image(array)
            (x, y),#top left cordinates
            (x + width, y + height),# bottom right
            (0, 255, 0),# BGRcolor(b=0,g=255,r=0)
            2# border thicknes
        )
        face=image_np[y:y+height,x:x+width]
      
      #cv2.rectangle -draws a rectangle on the image_np with the specified coordinates and color (green in this case) and thickness (2 pixels).
    if face is not None:
           return Image.fromarray(image_np),Image.fromarray(face)


    return Image.fromarray(image_np), None


    #first image uploaded
    #this is read by pil
    #then we convert imge into numpy array
    #cuz mediapipe works on numpyarray only
    #then mediapipes pre-trained face detection model detects the face 
    # and return bbox cordinates
    #then i convert these cordinates into pixels(x,y,width,height)
    #opencv takes these info and draws a rectangle on the image_np
    #then finnaly i convert numpy array back to pil image 
    # and displayit using streamlit