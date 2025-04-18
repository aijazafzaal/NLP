import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def remove_hashtags_urls_at(data):
    data = re.sub(r"URL", "", data)  # Remogit addve "URL"
    data = re.sub(r"@\w+", "", data)  # Remove @mentions
    data = re.sub(r"#\w+", "", data)  # Remove hashtags
    return data

def lowercasing(data):
    data = data.lower()  # Lowercasing
    return data

def expand_emojis(data):
    data = emoji.demojize(data)  # Convert emojis to text
    return data

def remove_stopwords(data):
    stop_words = set(stopwords.words("english"))  # Load English stopwords
    words = data.split()  # Tokenize text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return " ".join(filtered_words)

def stemming(data):
    words = data.split()  # Tokenize text
    stemmed_words = [PorterStemmer().stem(word) for word in words]  # Apply stemming
    return " ".join(stemmed_words)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)