import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("Start time: ")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#----------------------------------------Loading the data set--------------------------------------------#
df = pd.read_csv('train.csv', encoding='ISO-8859-1')

#----------------------------------------Precrocessing data ------------------------------------#

# Remove duplicate rows based on 'textID' and drop rows where 'text' or 'sentiment' is null
df.drop_duplicates(subset='textID', inplace=True)
df.dropna(subset=['text', 'sentiment'], inplace=True)
df = df[['text', 'sentiment']]

# Convert 'sentiment' to numerical values
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

#----------------------------------------Precrocessing the data (text cleaning)------------------------------------#

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

df['processed_text'] = df['text'].apply(preprocess_text)

# Initialize and apply TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

# Save the vectorizer and the feature matrix to disk for later use
vectorizer_filepath = 'tfidf_vectorizer.pkl'
matrix_filepath = 'X_tfidf.pkl'

with open(vectorizer_filepath, 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

with open(matrix_filepath, 'wb') as file:
    pickle.dump(X_tfidf, file)

print("Vectorizer and feature matrix have been saved to disk.")

#----------------------------------------Splitting the data set--------------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['sentiment'], test_size=0.2, random_state=42)

#----------------------------------------Naive Bayes classifier--------------------------------------------#

# Initialize the classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the sentiments on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to disk
model_filepath = 'sentiment_classifier_naive_bayes.pkl'
with open(model_filepath, 'wb') as file:
    pickle.dump(clf, file)

print("Trained model has been saved to disk.")

#----------------------------------------(Hyperparameter Tuning) Naive Bayes classifier--------------------------------------------#

# Define the parameter grid for Naive Bayes
param_grid_nb = {
    'alpha': [0.01, 0.1, 1, 10]
}

# Initialize the Grid Search model for Naive Bayes
grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_nb.fit(X_train, y_train)

# Best parameters found
print("Best Parameters for Naive Bayes:", grid_search_nb.best_params_)

# Predict the sentiments on the test set using the best estimator
y_pred_nb = grid_search_nb.best_estimator_.predict(X_test)

# Evaluate the model
print("Tuned Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Tuned Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# Save the best Naive Bayes model from hyperparameter tuning to disk
best_nb_model_filepath = 'best_naive_bayes_classifier.pkl'
with open(best_nb_model_filepath, 'wb') as file:
    pickle.dump(grid_search_nb.best_estimator_, file)
print("Best Naive Bayes model saved to disk.")

#----------------------------------------SVM classifier--------------------------------------------#
# Initialize the SVM classifier
svm_clf = SVC()

# Train the classifier
svm_clf.fit(X_train, y_train)

# Predict the sentiments on the test set
y_pred_svm = svm_clf.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Save the trained model to disk
with open('svm_classifier.pkl', 'wb') as file:
    pickle.dump(svm_clf, file)

#----------------------------------------(Hyperparameter Tuning) SVM classifier--------------------------------------------#
# Define the parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Initialize the Grid Search model for SVM
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_svm.fit(X_train, y_train)

# Best parameters found
print("Best Parameters for SVM:", grid_search_svm.best_params_)

# Predict the sentiments on the test set using the best estimator
y_pred_svm = grid_search_svm.best_estimator_.predict(X_test)

# Evaluate the model
print("Tuned SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Tuned SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Save the best SVM model from hyperparameter tuning to disk
best_svm_model_filepath = 'best_svm_classifier.pkl'
with open(best_svm_model_filepath, 'wb') as file:
    pickle.dump(grid_search_svm.best_estimator_, file)
print("Best SVM model saved to disk.")

#----------------------------------------Random Forest classifier--------------------------------------------#

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
rf_clf.fit(X_train, y_train)

# Predict the sentiments on the test set
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save the Random Forest model to disk
with open('random_forest_classifier.pkl', 'wb') as file:
    pickle.dump(rf_clf, file)

#----------------------------------------(Hyperparameter Tuning) Random Forest classifier--------------------------------------------#

# Parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Initialize the Grid Search model
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search_rf.fit(X_train, y_train)

# Best parameters found
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)

# Predict the sentiments on the test set using the best estimator
y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)

# Evaluate the model
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save the best Random Forest model from hyperparameter tuning to disk
best_rf_model_filepath = 'best_random_forest_classifier.pkl'
with open(best_rf_model_filepath, 'wb') as file:
    pickle.dump(grid_search_rf.best_estimator_, file)
print("Best Random Forest model saved to disk.")

#----------------------------------------Comparative Evaluation--------------------------------------------#
# Initialize a DataFrame to hold all evaluation metrics
metrics_summary = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Function to calculate metrics and return a new row
def get_metrics_row(classifier_name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    return {'Classifier': classifier_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1_score}

# Calculate metrics and update summary for each classifier
nb_row = get_metrics_row('Naive Bayes', y_test, y_pred_nb)
svm_row = get_metrics_row('SVM', y_test, y_pred_svm)
rf_row = get_metrics_row('Random Forest', y_test, y_pred_rf)

# Concatenate the rows into the metrics_summary DataFrame
metrics_summary = pd.concat([metrics_summary, pd.DataFrame([nb_row, svm_row, rf_row])], ignore_index=True)

# Print and save the metrics summary
print(metrics_summary)
metrics_summary.to_csv('model_performance_summary.csv', index=False)
print("Model performance summary has been saved to disk.")

print("End time: ")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))