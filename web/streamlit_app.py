import streamlit as st
import requests
from PIL import Image
import io


st.title("Plant Image Classification")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict"):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format)
        img_bytes = img_bytes.getvalue()
        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    
        result = response.json()
        predicted_class = result["prediction"]
        display_text = " ".join(predicted_class.split("_"))
        dynamic_link = f"https://www.google.com/search?q={display_text}"

        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;align-items:center;margin-top:20px;">
                <p>That's {display_text} with {result['confidence']} confidence.</p><br />
                <a href="{dynamic_link}" target="_blank" style="text-decoration:none;">
                    <button style="padding:10px 20px;background-color:#4CAF50;color:white;border:none;border-radius:5px;cursor:pointer;">
                        Check on Google for {display_text}
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )