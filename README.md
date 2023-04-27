# Image Analysis of Plant Based Meat Products

This repository contains code for a Streamlit interactive demo for a food image recognition model, specifically for classifying images of plant-based meat products. 

## Project Overview

Our project aimed to train a deep learning model that could accurately classify images of plant-based meat products. We started by collecting a dataset of images of various plant-based meat products, which we used to train our model. We then used transfer learning techniques to fine-tune a pre-trained ResNet50 model on our dataset.

The `models` directory contains two models: `qa_resnet50_finetuned.pt` and `resnet50_experiment.pt`. The `qa_resnet50_finetuned.pt` file is the fine-tuned model we trained on the Food101 dataset and then further trained on our collected images using transfer learning. The `resnet50_experiment.pt` file is our final trained model used for evaluation.

The `qa-reset50.ipynb` file contains the code for training the model. We used PyTorch as our deep learning framework.

## Usage

To use the trained model, you can run the Streamlit app contained in `app.py`. The app allows you to upload an image of a plant-based meat product and the model will predict the type of food it is. 

To run the Streamlit app, simply run the following command:

```
streamlit run app.py
```

All necessary dependencies are listed in `requirements.txt`.
