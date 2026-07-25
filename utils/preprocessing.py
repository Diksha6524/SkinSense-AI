# from PIL import Image
import numpy as np

def preprocess_image(image):

    image = image.resize((224, 224))#resize

    image = np.array(image)#conversion pil to numpy
    image = image / 255.0 # FOR NORMALIZATION
# divide by 255 cuz  Image pixels (0–255)
#std 8 bit image stores color channel in range 0 to 255
    image = np.expand_dims(image, axis=0)# OG 3 DIM WHERE PRESTENT height width channel,
    #but mobilenetv2 need 4 dim ie batchsize height width channels
    # as tensorflow  proceses i batches that is in one batch we can have one or more then 1 faces
    #therfore batchsixze need this batchsize is put on axis 0,axis1-height,so on
    return image