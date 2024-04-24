import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------------------Loading the test data set--------------------------------------------#
df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')

# ----------------------------------------Preprocessing data ------------------------------------#

# Remove duplicate rows based on 'textID' and drop rows where 'text' is null
df_test.drop_duplicates(subset='textID', inplace=True)
df_test.dropna(subset=['text'], inplace=True)
df_test = df_test[['text', 'sentiment']]

# Convert 'sentiment' to numerical values
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df_test['sentiment'] = df_test['sentiment'].map(sentiment_mapping)


# ----------------------------------------Preprocessing the data (text cleaning)------------------------------------#

# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{3,}', '<PROFANITY>', text)
    text = text.lower()
    negations = ["not", "n't", "no", "cannot", "never", "nothing", "nowhere", "noone", "none",
                 "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't", "wouldn't",
                 "don't", "doesn't", "didn't", "isn't", "aren't", "ain't"]
    for negation in negations:
        text = re.sub(r'\b' + negation + r' \w+', lambda match: match.group(0).replace(" ", "_"), text)
    text = re.sub(r'[^a-zA-Z_\s]', '', text)
    tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(text) if token not in stop_words and token.isalpha()]
    return ' '.join(tokens)

df_test['processed_text'] = df_test['text'].apply(preprocess_text)

# Load the vectorizer from disk and transform the test data
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

X_test_tfidf = tfidf_vectorizer.transform(df_test['processed_text'])

# Load the best models from disk and predict
classifiers = ['naive_bayes', 'svm', 'random_forest']
classifier_files = {
    'naive_bayes': 'best_naive_bayes_classifier.pkl',
    'svm': 'best_svm_classifier.pkl',
    'random_forest': 'best_random_forest_classifier.pkl'
}

for classifier_name in classifiers:
    with open(classifier_files[classifier_name], 'rb') as file:
        loaded_classifier = pickle.load(file)
        y_pred = loaded_classifier.predict(X_test_tfidf)
        df_test[f'{classifier_name}_pred'] = y_pred
        print(f"Accuracy for {classifier_name}: {accuracy_score(df_test['sentiment'], y_pred)}")
        print(f"Classification Report for {classifier_name}:\n{classification_report(df_test['sentiment'], y_pred)}")

# Save the dataframe with predictions to a new CSV file
df_test.to_csv('test_with_predictions.csv', index=False)

print("CSV file with classifier predictions has been saved.")

