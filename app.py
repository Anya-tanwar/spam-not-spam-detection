import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()

    words = nltk.word_tokenize(text)

    y = []

    for i in words:
        if i.isalnum():
            y.append(i)

    words = y[:]
    y.clear()

    for i in words:
        if i not in stopwords.words('english'):
            y.append(i)

    words = y[:]
    y.clear()

    for i in words:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(
    open("vectorizer.pkl","rb")
)

model = pickle.load(
    open("model.pkl","rb")
)

st.title("Spam Email Detector")

input_text = st.text_area(
    "Enter Email Message"
)

if st.button("Predict"):

    transformed = transform_text(
        input_text
    )

    vector = tfidf.transform(
        [transformed]
    )

    result = model.predict(vector)[0]

    if result == 1:
        st.error("Spam Email")
    else:
        st.success("Not Spam")