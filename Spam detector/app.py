import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords and punkt if you haven't already
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Load the pre-trained model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Model or vectorizer file not found. Please ensure they are saved correctly.")
    st.stop()  # Stop the app if files are not found
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop the app for any other errors


# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove punctuation and stopwords, and apply stemming
    y = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]

    return " ".join(y)


# Set the title of the app
st.title("Spam Detector")

# HTML and CSS for the decorated text box
st.markdown("""
    <style>
        .decorated-box {
            background-color: #FFD700  /* Pitch yellow color */
            border: 2px solid #FF9800;  /* Orange border */
            border-radius: 10px;  /* Rounded corners */
            padding: 20px;  /* Padding inside the box */
            text-align: center;  /* Center text */
            font-size: 20px;  /* Font size */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Shadow effect */
        }
    </style>
    <div class="decorated-box">
        <h2>Welcome to the Spam Detector!</h2>
        <p>Enter your message below to check if it's spam or not.</p>
    </div>
""", unsafe_allow_html=True)

# Text input area for user to enter the message
input_sms = st.text_area("Enter the message")

# Button for prediction
if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")