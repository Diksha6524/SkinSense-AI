import streamlit as st# for ui(buttons, text, images, etc)
from PIL import Image #used to import the image class from the python imaginglib(pil)(pillow)#open read process images

st.title("SkinSense AI")

st.write("Upload a facial image for skin analysis.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
#button created to upload and image 

if uploaded_file:#if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
