from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import io
import json
import time
app = FastAPI()
model = tf.keras.models.load_model('./final_trained_model.h5')

file_path = r'./production.json'
with open(file_path, 'r') as file:
    class_names_dict = json.load(file)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict/")
async def predict_species(file: UploadFile = File(...)):
    start_time = time.time()

    received_time = time.time()
    contents = await file.read()
    receiving_time = time.time() - received_time

    preprocess_start = time.time()
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(contents), target_size=(480, 480))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    preprocess_time = time.time() - preprocess_start

    prediction_start = time.time()
    prediction = model.predict(image)
    prediction_time = time.time() - prediction_start

    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names_dict[str(predicted_class_idx)]
    confidence = np.max(prediction) * 100

    total_time = time.time() - start_time

    return {
        "prediction": predicted_class_name,
        "confidence": confidence,
        "times": {
            "receiving": receiving_time,
            "preprocessing": preprocess_time,
            "prediction": prediction_time,
            "total": total_time
        }
    }