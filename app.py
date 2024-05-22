import streamlit as st
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import csv
import os
import wget

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
num_classes = 3580

# Download the model file from Dropbox
def download_model_from_dropbox():
    model_path = 'Aditya_Sengupta_ResNet18_LeAF_3580_20240522_114850.pth'
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        url = 'https://www.dropbox.com/scl/fi/g5ugkvcpmjcjt0blfitjn/Aditya_Sengupta_ResNet18_LeAF_3580_20240522_114850.pth?rlkey=iuztmenkcc59fh9724jpxjqzn&st=bz04akr1&dl=0'
        wget.download(url, model_path)
        st.write("Model downloaded successfully!")
    return model_path

# Load the pre-trained ResNet-18 model and modify the final layer
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Function to load and preprocess an image
def load_image(image):
    image = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Function to run inference on a single image and get the class name
def run_inference(image, model, class_names):
    model.eval()
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    return class_names[class_id]

# Download the model immediately when the app starts
model_path = download_model_from_dropbox()
model = load_model(model_path)

# Streamlit app
st.title('LeAF 3580 Pest Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load the class names from a CSV file
    class_names_csv_path = 'leaf-3580-pest-classes.csv'
    class_names = {}
    with open(class_names_csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header row
        for rows in reader:
            class_names[int(rows[0])] = rows[1]

    predicted_class_name = run_inference(uploaded_file, model, class_names)
    st.write(f'Predicted class for the image is: {predicted_class_name}')
