# Toxic Comment Classification

A deep learning project to classify toxic online comments into six categories. This repository contains a Jupyter Notebook to train a model using TensorFlow/Keras and an interactive Gradio demo.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)

## 📝 Project Overview

The goal of this project is to build an effective multi-label text classification model. Given a piece of text (a comment), the model predicts the probability of it belonging to each of six toxicity classes: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. This is a critical task for online content moderation.

## 🗂️ Dataset

This project uses the dataset from the **Jigsaw Toxic Comment Classification Challenge**.

-   `train.csv`: Contains over 150,000 comments with their corresponding labels.
-   `test.csv`: Contains comments for which predictions are to be made.
-   `test_labels.csv`: Contains the labels for the test data (some are marked as -1, meaning they are not used for scoring).

## ⚙️ Workflow

The project notebook (`Toxicity_Cleaned.ipynb`) follows these steps:

1.  **Setup**: Installs all required libraries like TensorFlow and Gradio.
2.  **Data Loading**: Loads the `train.csv` dataset using Pandas.
3.  **Preprocessing**:
    -   Separates the comment text (features) from the six label columns (targets).
    -   Uses a `TextVectorization` layer to create a standardized vocabulary and convert text sentences into integer sequences.
4.  **Data Pipeline**: Creates an efficient `tf.data.Dataset` pipeline to handle shuffling, batching, and prefetching for optimal training performance.
5.  **Model Building**: Defines a Keras `Sequential` model with Embedding, Bidirectional LSTM, and Dense layers.
6.  **Training**: Trains the model on the preprocessed data, using a validation split to monitor performance.
7.  **Evaluation**: Evaluates the model's accuracy, precision, and recall on a held-out test set.
8.  **Interactive Demo**: Launches a Gradio web interface to allow for real-time classification of custom text inputs.

## 🧠 Model Architecture

The neural network is built with the following layers:

1.  **Input Layer**: `TextVectorization` layer to process raw text.
2.  **Embedding Layer**: Converts integer sequences into dense vector representations.
3.  **Bidirectional LSTM Layer**: Captures context from both forward and backward directions in a sequence, which is crucial for understanding language.
4.  **Dense Layers**: Fully connected layers with `ReLU` activation for learning high-level patterns.
5.  **Output Layer**: A final `Dense` layer with 6 units and a `sigmoid` activation function to output a probability for each of the six toxicity classes.

## 🛠️ Technologies Used

-   Python
-   TensorFlow & Keras
-   Pandas
-   Scikit-learn
-   Gradio
-   Matplotlib
-   Google Colab

## 🚀 How to Run

Follow these steps to run the project yourself:

1.  **Download Files**: Make sure you have the `Toxicity_Cleaned.ipynb` notebook and the `train.csv` file.
2.  **Open in Google Colab**:
    -   Go to [colab.research.google.com](https://colab.research.google.com).
    -   Select `File` -> `Upload notebook...` and choose `Toxicity_Cleaned.ipynb`.
3.  **Upload Data**:
    -   In the Colab file browser on the left, click the "Upload to session storage" button.
    -   Select and upload `train.csv`.
4.  **Run the Code**:
    -   Execute the cells in the notebook from top to bottom.
    -   The first cell will install all dependencies. The last cell will launch the Gradio demo.

## 📂 File Structure
-   **Toxicity_Cleaned.ipynb:** The main Jupyter notebook with all the code.
-   **train.csv:** Training data with comments and labels.
-   **test.csv:** Test data with comments.
-   **test_labels.csv:** Labels for the test data.
-   **sample_submission.csv:** An example of the required submission.
