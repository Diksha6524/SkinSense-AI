import streamlit as st# for ui(buttons, text, images, etc)
from PIL import Image #used to import the image class from the python imaginglib(pil)(pillow)#open read process images
from utils.face_detection import detect_faces
from utils.preprocessing import preprocess_image
st.title("SkinSense AI")

st.write("Upload a facial image for skin analysis.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
#button created to upload and image 

if uploaded_file:#if uploaded_file is not None:
    image = Image.open(uploaded_file)

    processed_image,cropped_face = detect_faces(image)
<<<<<<< HEAD
    if cropped_face is not None:
        cropped_face = preprocess_image(cropped_face)
    
    

=======
>>>>>>> 5f941d962598875d5617284259ab217ede2b44b5

    st.image(
        processed_image,
        caption="Detected Face",
        use_container_width=True
    )


    if cropped_face is not None:
<<<<<<< HEAD
        st.image(
        cropped_face,
        caption="Resized Face(224 * 224)",
        use_container_width=True
     )
    else:
        st.warning("no face dettected")
=======
     st.image(
        cropped_face,
        caption="Cropped Face",
        use_container_width=True
     )
    else:
       st.warning("no face dettected")
>>>>>>> 5f941d962598875d5617284259ab217ede2b44b5
# to run this code--  python -m streamlit run app.py
