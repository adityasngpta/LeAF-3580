import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision import transforms
import csv
import gdown
import os

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
num_classes = 3580

# Load the pre-trained ResNet-18 model and modify the final layer
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Function to download the model from Google Drive
def download_model_from_drive():
    if not os.path.exists('model.pth'):
        st.write("Downloading model...")
        gdown.download('https://drive.google.com/file/d/19b40vxjQ3fXEIA65kNtPuvuzqdWjznVB/view?usp=share_link', 'model.pth', quiet=False)
        st.write("Model downloaded successfully!")

# Call the function to download the model
download_model_from_drive()

# Load the saved model checkpoint
checkpoint_path = 'model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)  # Use map_location to load on CPU

# Remove 'module.' prefix from the keys if the checkpoint was saved from a DataParallel model
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]  # remove 'module.' prefix
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Define the data transformation for inference
transform_inference = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Function to load and preprocess an image
def load_image(image):
    image = Image.open(image)
    image = transform_inference(image).unsqueeze(0)
    return image.to(device)

# Load the class names from a CSV file
def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header row
        for rows in reader:
            class_names[int(rows[0])] = rows[1]
    return class_names

# Function to run inference on a single image and get the class name
def run_inference(image, class_names):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    return class_names[class_id]

# Load the class names CSV file
class_names_csv_path = 'leaf-3580-pest-classes.csv'
class_names = load_class_names(class_names_csv_path)

# Streamlit app
st.title('LeAF 3580 Pest Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predicted_class_name = run_inference(uploaded_file, class_names)
    st.write(f'Predicted class for the image is: {predicted_class_name}')
