import tensorflow as tf #tf-laods daataset and trains model
from pathlib import Path# path-creates reliabel path to daatset
import matplotlib.pyplot as plt #plt-visualizes the training process


dataset_path = Path(__file__).resolve().parent.parent / "dataset"

train_dataset=tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,#validation datastes used for evaluating the model not for training
    subset="training",#subset of the data to use for training ie 80%
    seed=42,#Tis means your model is being tested on different images every time.
#ex-Training:
# 1 2 4 5 6 7 9 10
# Validation:
# 3 8
#why 42-It became popular because of the science-fiction book The Hitchhiker's Guide to the Galaxy, where 42 is "the answer to the ultimate question of life, the universe, and everything."
    image_size=(224, 224),#Resize all images---MobileNetV2 expects this size
    batch_size=32#Process 32 images at a time
)


#validation daataset
validation_dataset=tf.keras.utils.image_dataset_from_directory(
    dataset_path,   
    validation_split=0.2,
    subset="validation",#subset of the data to use for validation ie 20%
    seed=42,
    image_size=(224, 224),
    batch_size=32
)


# The numbers are called labels.

# The words are the class names.




print("\nDataset Loaded Successfully!")
print("-" * 40)

print("Class Names:")
print(train_dataset.class_names)

print("\nNumber of Classes:")
print(len(train_dataset.class_names))

print("\nTraining Batches:")
print(len(train_dataset))

print("\nValidation Batches:")
print(len(validation_dataset))




#for printing iamges -testing

for images, labels in train_dataset.take(2):#first batch--gets 32 images

    plt.figure(figsize=(10, 10))#Creates a figure large enough to display multiple images.

    for i in range(9):#only forshowing first 9 images

        plt.subplot(3, 3, i + 1)#3 X 3 grid

        plt.imshow(images[i].numpy().astype("uint8"))#converts tensorFlow tensor into numpy array-as mtplt can display only numpy arrays

        plt.title(train_dataset.class_names[labels[i]])#i takes number ie 1,2, or any ie it becmoes ex-[labels[0]]and  this becomez [5] ie [labels[0]]=5;;train_dataset.class_names[5]  it returns redness so title it gets is redness

        plt.axis("off")#removes axis so that only photo is displayed

    plt.show()


# whats tensor in TensorFlow-powerful version of numpy array



#data augmentation -Imagine teaching a child to recognize a cat.
# You don't show only one cat.
# You show:
#  White cat
#  Black cat
# Fat cat
# Small cat
# Sitting cat
# Standing cat
# Now the child understands:
# "They're all cats."


#that is altering data/ photo in such a way that they will be relaistic to the photos user will upload
#that is usermight upload photo which is slightly at angle ,zoomed in or under differnt lights etc

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),#it flips image left<-> right
    tf.keras.layers.RandomRotation(0.1),#fraction of full rotation ie 0.1*360=36->rotation is choosen randomly within that range (both clockwise and countercloclwise)
    tf.keras.layers.RandomZoom(0.1),#Randomly zooms in or out by about 10%.
    tf.keras.layers.RandomContrast(0.1)#Randomly changes image contrast slightly.
])



#So we're passing the image through the augmentation pipeline, and it returns a new, randomly modified version.-therfore we use   augmented_image = data_augmentation(image)






#for visulaization of data augmentation
for images, labels in train_dataset.take(1):

    plt.figure(figsize=(10, 10))

    first_image = images[0]

    for i in range(9):

        augmented_image = data_augmentation(
            tf.expand_dims(first_image, axis=0)#TensorFlow functions
        )
#expand_dims()-> adds back the batch of dimension ie before t was (224,224,3) now(1,224,224,3)1 is 1 image
#axis=0 fo adding batch dimension in the beggining
        plt.subplot(3, 3, i + 1)

        plt.imshow(augmented_image[0].numpy().astype("uint8"))

        plt.axis("off")

    plt.show()



#working flow

# Original Batch

# (32,224,224,3)
#         │
# images[0]
#         │
# (224,224,3)
#         │
# expand_dims()
#         │
# (1,224,224,3)
#         │
# Data Augmentation
#         │
# (1,224,224,3)
#         │
# augmented_image[0]
#         │
# (224,224,3)
#         │
# plt.imshow()