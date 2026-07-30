from pathlib import Path
from PIL import Image

dataset_path = Path(__file__).resolve().parent.parent / "dataset"

print("Checking every image...\n")

bad_files = []

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

for file in dataset_path.rglob("*"):

    if not file.is_file():
        continue

    if file.suffix.lower() not in valid_extensions:
        bad_files.append((file, "Invalid extension"))
        continue

    try:
        with Image.open(file) as img:
            img.load()   # Fully decode the image
    except Exception as e:
        bad_files.append((file, str(e)))

print(f"\nBad files found: {len(bad_files)}\n")

for file, error in bad_files:
    print(file)
    print("   ->", error)