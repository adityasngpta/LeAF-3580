import streamlit as st
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import csv
import gdown
import os
import zipfile

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
num_classes = 3580

# Download the ZIP archive containing the model file from Google Drive
def download_model_from_drive():
    model_zip_path = 'model.zip'
    if not os.path.exists(model_zip_path):
        st.write("Downloading model...")
        gdown.download('https://drive.google.com/uc?id=15VF_6pbTfWhMX8COB5REns1csJ25D3Ff', model_zip_path, quiet=False)
        st.write("Model downloaded successfully!")
    return model_zip_path

# Extract the model from the ZIP archive
def extract_model(model_zip_path):
    if os.path.isfile(model_zip_path) and model_zip_path.endswith('.zip'):
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        return 'model.pth'
    else:
        st.error("The downloaded file is not a valid ZIP archive.")

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

# Streamlit app
st.title('LeAF 3580 Pest Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    model_zip_path = download_model_from_drive()
    model_path = extract_model(model_zip_path)
    model = load_model(model_path)

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
