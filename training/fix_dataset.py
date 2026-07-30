from pathlib import Path
from PIL import Image

dataset = Path(__file__).resolve().parent.parent / "dataset"

count = 0

for img_path in dataset.rglob("*"):

    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

        try:
            img = Image.open(img_path)
            img = img.convert("RGB")
            img.save(img_path)

            count += 1

            if count % 100 == 0:
                print(f"Fixed {count} images...")

        except Exception as e:
            print("Couldn't fix:", img_path)
            print(e)

print("\nFinished!")
print("Total fixed:", count)