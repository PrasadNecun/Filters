import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from utils import *
from models.models import *
import os

app = FastAPI()

input_size = (256, 256, 1)

# Function to preprocess the image
def preprocess_image(image_path):
    deg_image = Image.open(image_path).convert('L')
    return deg_image

# Function to pad the image
def pad_image(image):
    h = ((image.shape[0] // 256) + 1) * 256
    w = ((image.shape[1] // 256) + 1) * 256
    padded_image = np.zeros((h, w)) + 1
    padded_image[:image.shape[0], :image.shape[1]] = image
    return padded_image, h, w

# Function to predict using the generator model
def predict_image(task, image):
    if task == "deblur":
        generator = generator_model(biggest_layer=1024)
        generator.load_weights("weights/deblur_weights.h5")
    elif task == "watermarkremoval":
        generator = generator_model(biggest_layer=512)
        generator.load_weights("weights/watermark_rem_weights.h5")
    else:
        raise ValueError("Wrong task, please specify a correct task!")

    test_image_p = split2(image.reshape(1, image.shape[0], image.shape[1], 1), 1, image.shape[0], image.shape[1])
    predicted_list = [generator.predict(test_image_p[l].reshape(1, 256, 256, 1)) for l in range(test_image_p.shape[0])]
    predicted_image = np.array(predicted_list)
    return predicted_image

@app.post("/process/")
async def process_image(task: str = Form(...), file: UploadFile = File(...)):
    # Save the uploaded file
    image_path = f"uploaded_{file.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Preprocess the image
    deg_image = preprocess_image(image_path)
    deg_image.save('preprocessed_image.png')
    
    test_image = plt.imread('preprocessed_image.png')
    
    # Pad the image
    padded_image, h, w = pad_image(test_image)
    
    # Predict the output image
    predicted_image = predict_image(task, padded_image)
    
    # Merge the image and crop to original size
    predicted_image = merge_image2(predicted_image, h, w)
    predicted_image = predicted_image[:test_image.shape[0], :test_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    
    # Save the predicted image
    save_path = "predicted_image.jpg"
    plt.imsave(save_path, predicted_image, cmap='gray')
    
    # Remove the temporary uploaded file
    os.remove(image_path)
    
    return FileResponse(save_path)

# To run the app, use the command:
# uvicorn script_name:app --reload
