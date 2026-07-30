from PIL import Image
from models.predictor import predict_skin

image = Image.open("test.jpg")

prediction, confidence = predict_skin(image)

print("Prediction :", prediction)
print("Confidence :", confidence)