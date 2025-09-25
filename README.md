Toxic Comment Classification
This project uses a deep learning model to classify online comments into six categories of toxicity: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The model is built with TensorFlow/Keras and includes an interactive web interface created with Gradio for real-time predictions.

📋 Table of Contents
Project Overview

Dataset

Workflow

Model Architecture

Technologies Used

Setup and Usage

File Structure

📝 Project Overview
The goal of this project is to build an effective multi-label text classification model. Given a piece of text (a comment), the model predicts the probability of it belonging to each of the six toxicity classes. This is a common and important task in moderating online platforms and fostering healthier online conversations.

🗂️ Dataset
The project uses the dataset from the Jigsaw Toxic Comment Classification Challenge.

train.csv: Contains the training comments along with their binary labels for each of the six toxicity categories.

test.csv: Contains comments to be classified.

test_labels.csv: Contains the labels for the test.csv data.

Each comment in train.csv has a binary value (0 or 1) for each of the following columns:

toxic

severe_toxic

obscene

threat

insult

identity_hate

⚙️ Workflow
The project follows these key steps, as implemented in Toxicity_Cleaned.ipynb:

Data Loading: The train.csv file is loaded into a Pandas DataFrame.

Data Preprocessing:

Features (comment text) and labels (toxicity categories) are separated.

A TextVectorization layer from Keras is used to clean the text, create a vocabulary, and convert text into integer sequences.

Data Pipeline: A tf.data.Dataset pipeline is created to efficiently handle data batching, shuffling, and prefetching, which optimizes performance during training.

Data Splitting: The dataset is split into training, validation, and test sets.

Model Building: A Sequential model is constructed using Keras.

Training: The model is trained on the training set while being monitored on the validation set to prevent overfitting.

Evaluation: The trained model's performance is evaluated on the unseen test set using metrics like Precision, Recall, and Categorical Accuracy.

Interactive Demo: A simple web interface is launched using Gradio, allowing users to input their own comments and get instant toxicity predictions.

🧠 Model Architecture
The neural network is a Sequential model with the following layers:

TextVectorization Layer: The input layer that preprocesses raw text.

Embedding Layer: Maps the integer-encoded vocabulary into dense vectors of a fixed size (32 dimensions).

Bidirectional LSTM Layer: A Long Short-Term Memory layer that processes sequences of text, capturing context from both forward and backward directions. It uses the tanh activation function.

Dense Layers: A series of fully-connected layers (ReLU activation) to learn complex patterns from the features extracted by the LSTM.

Output Layer: A final Dense layer with 6 units (one for each class) and a sigmoid activation function to output a probability between 0 and 1 for each toxicity category.

🛠️ Technologies Used
Python 3

TensorFlow / Keras: For building and training the deep learning model.

Pandas: For data manipulation and loading CSV files.

NumPy: For numerical operations.

Scikit-learn: Used for its utility functions.

Matplotlib: For plotting the training history.

Gradio: For creating the interactive web demo.

Google Colab: As the environment for running the notebook.

🚀 Setup and Usage
To run this project, follow these steps:

Clone or Download: Get the project files onto your local machine.

Open in Google Colab:

Go to colab.research.google.com.

Click on File > Upload notebook and select the Toxicity_Cleaned.ipynb file.

Upload Data:

In the Colab interface, click the folder icon on the left sidebar to open the file browser.

Click the "Upload to session storage" icon and upload your train.csv file.

Run the Notebook:

Execute the cells in the notebook sequentially from top to bottom by pressing Shift + Enter in each cell.

The first code cell will install all necessary dependencies.

The final cell will launch a public Gradio link that you can open in your browser to interact with the model.

📂 File Structure
.
├── Toxicity_Cleaned.ipynb      # The main Jupyter notebook with all the code.
├── train.csv                   # Training data with comments and labels.
├── test.csv                    # Test data with comments.
├── test_labels.csv             # Labels for the test data.
├── sample_submission.csv       # An example of the required submission format for Kaggle.
└── README.md                   # This file.
