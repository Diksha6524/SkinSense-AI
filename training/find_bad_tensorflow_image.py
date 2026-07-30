from pathlib import Path
import tensorflow as tf

dataset_path = Path(__file__).resolve().parent.parent / "dataset"

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

print("Checking images with TensorFlow...\n")

bad = []

for file in dataset_path.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() not in valid_extensions:
        continue

    try:
        image = tf.io.read_file(str(file))
        tf.io.decode_image(image)

    except Exception as e:
        print(f"\n❌ Bad image:")
        print(file)
        print(e)
        bad.append(file)

print("\n--------------------------------")
print("Total bad images:", len(bad))
