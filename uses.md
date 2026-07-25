# open cv-
low-level image processing.
It operates on images as NumPy arrays in the BGR (Blue, Green, Red) color format.
helps with-
Reading images
Writing images
Drawing shapes
Face detection
Video processing
Color conversion
ex-img.shape
(480, 640, 3)
Height = 480 pixels
Width = 640 pixels
3 color channels


# mediapipe-(readymade pretrained AI model)
open-source framework.
Pre-trained Machine Learning models and pipelines.(know for its ready to use Pre-trained AI models)
helps with-
Face Detection
Face Mesh
Hand Tracking
Pose Estimation

# PIL(pillow)-used for manipulating ,reading,saving images in different file format.
Pythonic imaging library designed to read, manipulate, and save many different image file formats. 
uses RGB color space


# working together-
1. OpenCV is used to capture a live webcam feed.
2. The OpenCV BGR image is converted to RGB (often using OpenCV itself) and sent to
MediaPipe to detect body landmarks (e.g., hand or pose tracking).
3. The spatial data is processed, and PIL might be used to load custom graphics or text, which is then drawn back onto the OpenCV frame for display.


# digital image are array of pixels, each pixel made up of 3colors (red,green,blue) each color channel takes upto 1 byte of memory,holding value from 0 to 255

##  BGR-used by  cv,it reads blue byte first ,then green, then red
BGR representation: [0, 0, 255]
## RGB-expected input image by media pipeline(mp),it reads red byte, then green, then blue
RGB representation: [255, 0, 0]




# to convert the frame from cv(uses BGR) to mp(uses RGB) we need to convert captured BGR (by cv )to RGB (for mp)

1. Capture frame from webcam (Outputs BGR)
cap = cv2.VideoCapture(0)
success, frame_bgr = cap.read()

2. Convert to RGB so MediaPipe can read it properly
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

3. Process with MediaPipe
results = your_mediapipe_model.process(frame_rgb)