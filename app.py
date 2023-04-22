import streamlit as st
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained PyTorch model
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU

else:
    device = torch.device('cpu')   # Use CPU

model = torch.load('resnet50_experiment.pt', map_location=device)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


# print(device)


def app():
    # Set the title of the app
    #st.set_page_config(page_title="My Streamlit App")
    st.set_page_config(page_title="Image Analysis", page_icon=None, layout="centered",
                       initial_sidebar_state="auto")
    st.title("Image Analysis of Plant Based Meat Products")

    # Add a file uploader to get an image from the user
    uploaded_file = st.file_uploader(
        "Choose an image...", type=['jpg', 'jpeg', 'png'])

    # If an image has been uploaded, display it and make a prediction
    if uploaded_file is not None:
        # Open the image using PIL
        model.eval()
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = test_transform(image)
        preprocessed_image = torch.unsqueeze(preprocessed_image, 0)
        # Make a prediction using the model
        preprocessed_image = preprocessed_image.to(device)
        LABELS = [('Commercial', 'Fresh', 'Air Fry', 'Normal'), ('Commercial', 'Fresh', 'Air Fry', 'Over'), ('Commercial', 'Fresh', 'Deep Fry', 'Normal'), ('Commercial', 'Fresh', 'Deep Fry', 'Over'), ('Commercial', 'Fresh', 'NA', 'Unbaked'), ('Inhouse', 'Fresh', 'Air Fry', 'Normal'), ('Inhouse', 'Fresh', 'Air Fry', 'Over'),
                  ('Inhouse', 'Fresh', 'Deep Fry', 'Normal'), ('Inhouse', 'Fresh', 'Deep Fry', 'Over'), ('Inhouse', 'Old', 'Air Fry', 'Normal'), ('Inhouse', 'Old', 'Air Fry', 'Over'), ('Inhouse', 'Old', 'Deep Fry', 'Normal'), ('Inhouse', 'Old', 'Deep Fry', 'Over'), ('Inhouse', 'Fresh', 'NA', 'Unbaked')]

        prediction = model(preprocessed_image)
        # Get the index of the predicted class
        # predicted_class_index = prediction.argmax().item()
        _, predicted_class_index = torch.max(prediction.data, 1)
        # Get the label for the predicted class

        predicted_label = LABELS[predicted_class_index]

        # Display the image and the predicted class

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Product:', predicted_label[0], label_visibility="visible")
        col2.metric(
            'Fresh/Old:', predicted_label[1], label_visibility="visible")
        col3.metric(
            'Cooking Proccess:', predicted_label[2], label_visibility="visible")
        col4.metric('Level: ', predicted_label[3], label_visibility="visible")

        st.image(image, caption='Uploaded Image', use_column_width=True)


# Run the app
if __name__ == '__main__':
    app()
