import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# get word index
word_index = imdb.get_word_index()
# reverse the key, value of the word index above
reversed_word_index = {value: key for key, value in word_index.items()}

try:
    # load model
    model = load_model('simple_rnn_imdb.h5')
except Exception as e:
    st.error(f"Error loading model")
    st.stop()

# output dimension
embedding_dim = 128
# size of sentence
sentence_length = 500
vocab_size = 10_000

# get the word attached to each encoded index
def decode_review(encoded_review):
    return " ".join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# preprocess the given text to get it in the right shape for training
def preprocessed_text(text):
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Check for words not in vocabulary
    problem_words = [word for word in words if word_index.get(word, 2) + 3 >= vocab_size]
    if problem_words:
        st.error(f"The following words are not in our vocabulary: {', '.join(problem_words)}")
        st.error("Please try using different words or simpler language.")
        return None
        
    padded_review = sequence.pad_sequences([encoded_review], sentence_length)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocessed_text(review)
    if preprocessed_input is None:
        return None, None
    
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

### Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

# user input
user_input = st.text_input("Movie Review")

if user_input:
    # button to press to trigger review classification
    if st.button("Classify"):
        sentiment, prediction_score = predict_sentiment(user_input)
        if sentiment is not None:
            st.write("Predicted Sentiment:", sentiment)
            st.write(f"Confidence Score: {prediction_score:.4f}\n")
elif user_input == "":
    st.write("Please enter a review")
