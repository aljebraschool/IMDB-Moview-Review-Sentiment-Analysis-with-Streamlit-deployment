# IMDB-Moview-Review-Sentiment-Analysis-with-Streamlit-deployment

This repository implements a sentiment analysis model for IMDB movie reviews. The model classifies reviews as positive or negative, and the project features a Streamlit-based web application for user interaction.

## Features
- Sentiment analysis of IMDB movie reviews using a pre-trained RNN model.
- Web-based interactive interface for entering reviews and viewing predictions.
- Streamlit deployment for seamless accessibility.

## Demo
Access the deployed app here: [IMDB Sentiment Analysis App](https://imdb-moview-review-sentiment-analysis-with-app-deployment-mg4o.streamlit.app/).

## **Below is a demonstration of the app's interface:**
![image](https://github.com/user-attachments/assets/32066a64-d5af-49b5-a0f0-7e1399a02501)

## Repository Structure
- main.py: The Python script for the Streamlit application.
- simple_rnn_imdb.h5: Pre-trained RNN model in HDF5 format for performing sentiment analysis.
- requirements.txt: Python dependencies required for this project.
- prediction.ipynb: Jupyter Notebook for testing and validating the model.
- simple_rnn.ipynb: Jupyter Notebook for training the RNN model.
- README.md: Documentation for the project.

## How to Use
1. Visit the App Open the Streamlit app using this link: IMDB Sentiment Analysis App.

2. Input a Movie Review
- Type a movie review in the text box.
- Example: "The movie was an amazing experience with brilliant acting!"

3. Analyze Sentiment
- Click the "Classify" button.
- The app will classify the review as either positive or negative.

## Local Setup Instructions

To run this project locally, follow these steps:

1. Clone the Repository
   ``` bash
   git clone https://github.com/aljebraschool/IMDB-Moview-Review-Sentiment-Analysis-with-Streamlit-deployment.git
    cd IMDB-Moview-Review-Sentiment-Analysis-with-Streamlit-deployment
   ```
2. Install Dependencies
  Make sure you have Python installed. Then, install the required dependencies:
``` bash
pip install -r requirements.txt

```
3. Run the Application
  Start the Streamlit app:
``` bash
streamlit run app.py
```
The app will open in your browser, and you can interact with it locally.

## Model Description
The sentiment analysis model is built using a Recurrent Neural Network (RNN) and is trained on the IMDB movie reviews dataset. It consists of:
- Embedding Layer: Converts words into dense vectors.
- RNN Layer: Captures sequential dependencies in the review text.
- Output Layer: Classifies sentiment as either positive or negative.

## Dependencies

- Python 3.7+
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pickle

For a complete list of dependencies, see the requirements.txt file.

### License

This project is licensed under the MIT License.

### Contributing

Contributions are welcome! If you'd like to improve the project or fix any issues, please feel free to submit a pull request.

### Author

This project was created by [Algebra School](https://aljebraschool.github.io/).


