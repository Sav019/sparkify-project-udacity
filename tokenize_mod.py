import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize(text):

    """
    Tokenizes the input text by removing URLs, stripping non-alphabetic characters, 
    tokenizing the text into words, lemmatizing the words, and removing stop words.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        list: The list of cleaned tokens.
    """

    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    rest_regex = r"[^a-zA-Z0-9]"
    
    #Stripping messages of all urls ?
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #Stripping the messages of all symbols like ., or ?
    stripped = re.sub(rest_regex," ",text)
    # Tokenize the sentences to words
    tokens = word_tokenize(stripped)
    
    # Lemmatize the words (e.g. defined to define)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        #Remove Stop-words
        if tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)
        
    return clean_tokens