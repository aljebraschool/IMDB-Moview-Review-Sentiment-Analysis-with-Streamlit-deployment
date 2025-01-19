import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb # imdb is a movie review dataset that contains the revies of movies from the internet

#get word index
word_index = imdb.get_word_index()

#reverse the key, value of the word index above
reversed_word_index = {value : key for key, value in word_index.items()}

#load model
model = load_model('simple_rnn_imdb.h5')


#output dimension
embedding_dim = 128

#size of sentence
sentence_length = 500

vocab_size = 10_000

#get the word attached to each encoded index
def decode_review(encoded_review):
    return " ".join([reversed_word_index.get(i - 3, '?')  for i in encoded_review])

#preprocess the given text to get it in the right shape for training
def preprocessed_text(text):
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], sentence_length)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocessed_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = 'positive' if prediction[0][0] > 0.5 else 'nagative'

    return sentiment, prediction[0][0]

### Stramlit App
st.title("IMDB Moview Review Sentiment Analysis")
st.write("Enter a movie review to classify it as possitive or nagative")

#user input
user_input = st.text_input("Movie Review")

if user_input:
    #button to press to trigger review classification
    if st.button("Classify"):
        sentiment, prediction_score = predict_sentiment(user_input)

        st.write("Predicted Sentiment : ", sentiment)
        st.write(f"Confidence Score : {prediction_score:.4f}\n")

else:
    st.write("Please enter a review")


