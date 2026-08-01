import tensorflow as tf
import numpy as np


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from pathlib import Path

# Load model only once
MODEL_PATH = Path(__file__).resolve().parent / "model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


CLASS_NAMES = [
    "blackheads",
    "dark_spots",
    "inflammatory_acne",
    "pigmentation",
    "pores",
    "redness",
    "whiteheads",
    "wrinkles"
]

#prediction
def predict_skin(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy
    image = np.array(image)

    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # MobileNet preprocessing
    image = preprocess_input(image.astype(np.float32))

    # Predict
    prediction = model.predict(image, verbose=0)[0]

    # Best prediction
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(prediction[predicted_index])

    # Top 3 predictions
    top3_indices = np.argsort(prediction)[::-1][:3]

    top3_predictions = []

    for idx in top3_indices:
        top3_predictions.append(
            (
                CLASS_NAMES[idx],
                float(prediction[idx])
            )
        )

    return predicted_class, confidence, top3_predictions

def get_severity(confidence):

    if confidence < 0.50:
        return "🟢 Mild"

    elif confidence < 0.80:
        return "🟡 Moderate"

    else:
        return "🔴 Severe"