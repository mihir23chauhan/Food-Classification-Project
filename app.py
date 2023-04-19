import streamlit as st
import torch
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
    st.title("Image Classification")

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
        LABELS = ['commercial_air_normal', 'commercial_air_over', 'commercial_deep_normal', 'commercial_deep_over', 'commercial_unbaked', 'inhouse_air_normal', 'inhouse_air_over',
                  'inhouse_deep_normal', 'inhouse_deep_over', 'inhouse_old_air_normal', 'inhouse_old_air_over', 'inhouse_old_deep_normal', 'inhouse_old_deep_over', 'inhouse_unbaked']

        prediction = model(preprocessed_image)
        # Get the index of the predicted class
        # predicted_class_index = prediction.argmax().item()
        _, predicted_class_index = torch.max(prediction.data, 1)
        # Get the label for the predicted class

        predicted_label = LABELS[predicted_class_index]

        # Display the image and the predicted class
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Prediction: ', predicted_label)


# Run the app
if __name__ == '__main__':
    app()
