from collections import Counter
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def csv_to_df(file_address: str, columns: list, encoding = 'utf-8') -> pd.DataFrame:
    try:
        df = pd.read_csv(file_address, encoding=encoding)
        df.columns = columns
        return df
    except FileNotFoundError:
        print(f"File '{file_address}' not found.")
    except pd.errors.EmptyDataError:
        print(f"File '{file_address}' is empty or has no data.")
    except pd.errors.ParserError:
        print(f"Error parsing CSV file '{file_address}'.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file '{file_address}': {str(e)}")

    return pd.DataFrame()


def eda_basic_assessment(df: pd.DataFrame) -> str:
    '''
        Check the shape of the dataset

        Check for missing values

        Explore unique sentiment categories

        Calculate the distribution of sentiment categories

        Compute the average length of news articles

        Check the data types of each column

        Determine the number of unique words or vocabulary size in the news column

        Calculate the percentage distribution of sentiment categories

    '''
    assessment = 'Basic EDA Assessment:\n'

    assessment += f'Dataset shape: {df.shape}\n---\n'

    assessment += f'Number of missing values: {df.isnull().sum()}\n---\n'

    assessment += f'Unique sentiment categories: {df["sentiment"].unique()}\n---\n'

    assessment += f'Sentiment category distribution: {df["sentiment"].value_counts()}\n---\n'

    article_length = df['news'].str.len()
    avg_len = article_length.mean()
    assessment += f'Average article length: {avg_len}\n---\n'

    assessment += f'Data Types: {df.dtypes}\n---\n'

    unique_words = set(" ".join(df['news']).split())
    vocab_size = len(unique_words)
    assessment += f'Vocabulary size: {vocab_size}\n---\n'

    sentiment_percentage = df['sentiment'].value_counts(normalize=True) * 100
    assessment += f'Sentiment category percentage{sentiment_percentage}\n---\n'

    return assessment


def eda_advanced_assessment(df: pd.DataFrame) -> str:
    '''
        Analyze the relationship between the length of news articles and sentiments

        Investigate the presence of outliers in the sentiment or news columns

        Identify and analyze the most common words associated with each sentiment category
    '''
    assessment = 'Advanced Assessment: \n'

    df['article length'] = df['news'].apply(len)
    sentiment_length = df.groupby('sentiment')['article length'].mean()
    assessment += f'Average article length by sentiment:\n{sentiment_length}\n---\n'

    news_length_outliers = df[np.abs(df['article length'] - df['article length'].mean()) > 3 * df['article length'].std()]
    assessment += f'News length outliers:\n{news_length_outliers}\n---\n'

    assessment += f'Most common words by sentiment:'
    sentiment_words = df.groupby('sentiment')['news'].apply(lambda x: " ".join(x)).reset_index()
    for index, row in sentiment_words.iterrows():
        sentiment = row['sentiment']
        words = row['news'].split()
        words_count = Counter(words)
        common_words = words_count.most_common(10)
        assessment += f'\nFor {sentiment} sentiment:\n'
        for word, count in common_words:
            assessment += f'{word}: {count}, '
    assessment += '\n--\n'


    return assessment


def perform_eda(df: pd.DataFrame) -> None:
    basic_assessment = eda_basic_assessment(df)

    advanced_assessment = eda_advanced_assessment(df)

    assessment = basic_assessment + advanced_assessment

    print(assessment)


def clean_text_parallel(text):
    # Check if the input is a pandas Series
    if isinstance(text, pd.Series):
        # Convert the Series to a list
        text = text.tolist()

    # Process each text element
    cleaned_texts = []
    for t in tqdm(text, desc='Processing', leave=False):
        # Text lowercasing
        t = t.lower()

        # Tokenization
        words = nltk.word_tokenize(t)

        # Stop word removal
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Join tokens back to text
        t = ' '.join(words)

        # Spelling correction
        t = str(TextBlob(t).correct())

        # Special character removal
        t = re.sub(r'[^A-Za-z\s]', '', t)

        # Tokenization after special character removal
        words = nltk.word_tokenize(t)

        # Remove numbers and numerical tokens
        words = [word for word in words if not word.isdigit()]

        # Join tokens back to text
        t = ' '.join(words)

        cleaned_texts.append(t)

    return cleaned_texts


def perform_cleaning_parallel(df: pd.DataFrame):
    # Number of processes to use (adjust according to available CPU cores)
    num_processes = 4

    # Split dataframe into chunks for parallel processing
    df_chunks = np.array_split(df, num_processes)

    # Create a multiprocessing pool
    pool = Pool(num_processes)

    # Apply cleaning function in parallel on each chunk
    cleaned_chunks = pool.map(clean_text_parallel, [chunk['news'] for chunk in df_chunks])

    # Concatenate the cleaned chunks back into a single list
    cleaned_texts = sum(cleaned_chunks, [])

    # Update the 'news' column in the original dataframe
    df['news'] = cleaned_texts

    return df


def split_data(df: pd.DataFrame):
    X = df['news']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def build_model():
    # Create the model
    nb_model = MultinomialNB()
    svm_model = SVC()

    return nb_model, svm_model


def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)

    # Perform model tuning if needed
    # Adjust hyperparameters, such as C or alpha values, based on grid search or cross-validation
    pass


def evaluate_model(model, X_test, y_test, label_encoder):
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Decode the predicted labels
    y_pred = label_encoder.inverse_transform(y_pred)
    y_test = label_encoder.inverse_transform(y_test)

    # Print classification report
    report = classification_report(y_test, y_pred)
    print(report)


def save_model(model, filename):
    # Save the trained model
    joblib.dump(model, filename)


def run():
    input_file = ''
    
    columns = ['sentiment', 'news']

    encoding = 'latin-1'

    df = csv_to_df(input_file, columns, encoding)

    perform_eda(df)

    clean_df = perform_cleaning_parallel(df)

    perform_eda(clean_df)

     # Split the data
    X_train, X_test, y_train, y_test = split_data(clean_df)

    # Encode sentiment labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Perform text vectorization
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Build models
    nb_model, svm_model = build_model()

    # Train models
    train_model(nb_model, X_train_vectorized, y_train_encoded)
    train_model(svm_model, X_train_vectorized, y_train_encoded)

    # Evaluate models
    print("Naïve Bayes Model Evaluation:")
    evaluate_model(nb_model, X_test_vectorized, y_test_encoded, label_encoder)
    print("\nSVM Model Evaluation:")
    evaluate_model(svm_model, X_test_vectorized, y_test_encoded, label_encoder)

    Save models
    save_model(nb_model, "naive_bayes_model.joblib")
    save_model(svm_model, "svm_model.joblib")


if __name__ == '__main__':
    run()
