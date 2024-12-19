import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download("stopwords")


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam.pkl", "rb"))

st.title("SMS Spam Detection")
spam_msg = st.text_input("enter your sms")


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    corpus = []

    for i in text:
        if (
            i.isalnum()
            and i not in stopwords.words("english")
            and i not in string.punctuation
        ):
            corpus.append(ps.stem(i))

    return " ".join(corpus)


# preprocess
processed_text = transform_text(spam_msg)
# Vectorize

vector = tfidf.transform([processed_text])

# Predict

result = model.predict(vector)[0]

if st.button("Predict", type="primary"):
    if result == 1:
        st.header("This message is a SPAM")
    else :
        st.header("This message is a Not SPAM")