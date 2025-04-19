# -IMDB-Movie-Reviews-Sentiment-Analysis
-----------------------------------------

This project focuses on binary sentiment classification using a dataset of IMDB movie reviews. The goal is to build a deep learning model that can predict whether a movie review is positive or negative based on its textual content. It demonstrates the use of NLP techniques and LSTM neural networks for effective sentiment analysis.

💡 Project Objectives
-----------------------

Build a model capable of understanding the sentiment behind textual data (movie reviews).
Apply preprocessing techniques to prepare raw text data for machine learning.
Train an LSTM-based deep learning model for accurate classification.
Evaluate and visualize the performance of the model.
Provide functionality for real-time sentiment prediction using custom input text.

🗃️ Dataset
------------

Source: IMDB Large Movie Review Dataset.
Size: 50,000 reviews.
Classes: Binary (positive, negative).

🔧 Tools and Technologies Used
-------------------------------

Python – Core programming language.
Pandas, NumPy – For data manipulation and numerical operations.
NLTK (Natural Language Toolkit) – For tokenization, stopword removal, etc.
TensorFlow and Keras – To build and train the LSTM model.
Matplotlib, Seaborn – For data visualization and performance plots.

⚙️ Key Steps and Components
----------------------------

Data Cleaning & Preprocessing
=============================
Lowercasing, removing HTML tags, punctuation, and stopwords.
Tokenizing and padding sequences for uniform input length.

Model Building
==============
LSTM (Long Short-Term Memory) layers for sequence modeling.
Embedding layer to convert tokens into dense vectors.
Dropout and Dense layers to prevent overfitting and improve learning.

Model Training & Evaluation
===========================
Trained using binary cross-entropy loss and Adam optimizer.
Accuracy, loss curves, and confusion matrix used for performance evaluation.

Prediction Interface
====================
Gradio web interface has been integrated.
Accepts user input and predicts sentiment in real-time.
Useful for deployment or integration into applications.

📈 Results
------------

The LSTM model demonstrated strong performance on the validation set, accurately capturing sentiment in a wide range of reviews.
