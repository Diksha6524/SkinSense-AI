import os
from pathlib import Path

dataset_path = Path(__file__).resolve().parent.parent / "dataset"
#path(__file__) gives the path of the current file,->training/dataset_info.py
#.resolve() gives the absolute path, 
# .parent.parent goes two levels up in the directory structure, and / "dataset" appends the "dataset" folder to the path.       #.parent->training/ and  .parent.parent->SkinSense-AI/
# overall it becomes ->SkinSense-AI/dataset

classes= sorted(os.listdir(dataset_path))  # this creates a list of all the classes in the dataset directory
#sorted()-used for sorting the list of classes in alphabetical order.
print("Dataset Summary")
print(classes)
total_images = 0
print("-"*40)
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    # print(class_path)
    images_per_class=len(os.listdir(class_path))
    # print(f"{class_name:<25} : {image_count}")
    print(f"{class_name:<25} : {images_per_class}")

    total_images += images_per_class
print("-"*40)
print(f"{ 'Total Images:':<25} : {total_images}")








#     print("\nDataset Summary")
# print("-" * 35)

# total_images = 0

# for class_name in classes:
#     class_path = os.path.join(dataset_path, class_name)

#     image_count = len(os.listdir(class_path))

#     print(f"{class_name:<25} : {image_count}")

#     total_images += image_count

# print("-" * 35)
# print(f"Total Images: {total_images}")