import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    stems = [ps.stem(token) for token in filtered_tokens]
    return " ".join(stems)
