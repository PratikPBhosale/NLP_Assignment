import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nDataset Sample:")
print(df.head())

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    
    # Apply BOTH stemming + lemmatization (to show learning outcome)
    words = [stemmer.stem(w) for w in words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

print("\nCleaned Text Sample:")
print(df[['text', 'clean_text']].head())

# -------------------------------
# Vectorization (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model 1: Naive Bayes
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -------------------------------
# Model 2: Logistic Regression
# -------------------------------
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -------------------------------
# Confusion Matrix
# -------------------------------
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))