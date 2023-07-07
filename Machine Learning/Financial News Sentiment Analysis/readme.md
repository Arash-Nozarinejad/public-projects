# Financial News Sentiment Analysis

This project aims to perform sentiment analysis on financial news articles using Python. The goal is to analyze the sentiment expressed in these articles and predict whether the overall sentiment is positive, negative, or neutral. The project utilizes two popular machine learning models, namely Naïve Bayes and Support Vector Machines (SVM), to accomplish this task.

- [Financial News Sentiment Analysis](#financial-news-sentiment-analysis)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Pre-cleaning EDA](#pre-cleaning-eda)
  - [Data Cleaning](#data-cleaning)
  - [Post-cleaning EDA](#post-cleaning-eda)
  - [Splitting the Data](#splitting-the-data)
  - [Building the Model](#building-the-model)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Saving the Model](#saving-the-model)
  - [Conclusion](#conclusion)

## Introduction

Sentiment analysis, also known as opinion mining, is a technique that involves analyzing textual data to determine the sentiment expressed within it. In the context of financial news, sentiment analysis can be a valuable tool for investors, traders, and financial institutions to gauge market sentiment and make informed decisions.

This project follows a step-by-step approach to perform sentiment analysis on a financial news dataset. The process involves data acquisition, exploratory data analysis (EDA), data cleaning and preprocessing, model creation, model training, evaluation, and comparison.

The primary focus of this project is to build and compare the performance of two machine learning models: Naïve Bayes and SVM. These models are widely used for sentiment analysis tasks and have shown promising results in various natural language processing applications.

The project aims to provide a comprehensive overview of the entire workflow, from data preprocessing to model evaluation. By following the project, readers will gain insights into the steps involved in sentiment analysis and understand how to apply machine learning techniques to financial news datasets.

Next, let's explore the dataset used in this project and where it can be downloaded.

## Dataset

The dataset used for this project is sourced from Kaggle and can be accessed at this [link](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news). Additionally, the dataset is also available in the `dataset` subdirectory of this GitHub repository.

The dataset consists of financial news articles along with their corresponding sentiment labels. The sentiment labels are categorized as positive, negative, or neutral, representing the sentiment expressed in each article. The dataset provides valuable insights into the sentiment associated with financial news, which can be useful for sentiment analysis and prediction tasks.

The dataset provides a varied range of financial news articles along with their corresponding sentiment labels.

## Pre-cleaning EDA

An exploratory data analysis (EDA) was performed on the dataset to gain insights and understand its characteristics before proceeding with data cleaning and preprocessing. The following code snippet combines the three EDA functions into one comprehensive analysis:

``` python
import pandas as pd
from collections import Counter

def perform_eda(df: pd.DataFrame) -> None:
    # Basic Assessment
    assessment = 'Pre-cleaning EDA Results:\n\n'

    # Dataset shape
    assessment += f"1. Dataset Shape: {df.shape}\n\n"

    # Number of missing values
    assessment += f"2. Missing Values:\n{df.isnull().sum()}\n\n"

    # Unique sentiment categories
    assessment += f"3. Unique Sentiment Categories: {df['sentiment'].unique()}\n\n"

    # Sentiment category distribution
    assessment += f"4. Sentiment Category Distribution:\n{df['sentiment'].value_counts()}\n\n"

    # Average article length
    df['article_length'] = df['news'].apply(len)
    avg_article_length = df['article_length'].mean()
    assessment += f"5. Average Article Length: {avg_article_length}\n\n"

    # Data types
    assessment += f"6. Data Types:\n{df.dtypes}\n\n"

    # Vocabulary size
    words = " ".join(df['news']).split()
    vocab_size = len(set(words))
    assessment += f"7. Vocabulary Size: {vocab_size}\n\n"

    # Sentiment category percentage
    sentiment_percentage = (df['sentiment'].value_counts(normalize=True) * 100).round(2)
    assessment += f"8. Sentiment Category Percentage:\n{sentiment_percentage}\n\n"

    # Advanced Assessment - Average article length by sentiment
    sentiment_avg_length = df.groupby('sentiment')['article_length'].mean()
    assessment += "Advanced Assessment:\n"
    assessment += f"9. Average Article Length by Sentiment:\n{sentiment_avg_length}\n\n"

    # Most common words by sentiment
    assessment += "10. Most Common Words by Sentiment:\n"
    sentiment_words = df.groupby('sentiment')['news'].apply(lambda x: " ".join(x)).reset_index()
    for index, row in sentiment_words.iterrows():
        sentiment = row['sentiment']
        words = row['news'].split()
        word_count = Counter(words)
        common_words = word_count.most_common(10)
        assessment += f"For {sentiment} sentiment:\n"
        for word, count in common_words:
            assessment += f"{word}: {count}, "
        assessment += '\n'
    
    print(assessment)

# Perform pre-cleaning EDA assessment
perform_eda(df)
```

__Here's an explanation of each step:__

- Dataset Shape: This displays the shape of the dataset, indicating the number of rows and columns.

- Missing Values: The number of missing values in each column is computed using the isnull().sum() function.

- Unique Sentiment Categories: The unique sentiment categories present in the "sentiment" column are extracted.

- Sentiment Category Distribution: The distribution of sentiment categories is calculated using value_counts().

- Average Article Length: The average length of the news articles is calculated by applying the len() function to each article and then computing the mean.

- Data Types: This section displays the data types of each column in the dataset.

- Vocabulary Size: The total number of unique words (vocabulary size) present in the "news" column is determined.

- Sentiment Category Percentage: The percentage distribution of sentiment categories is calculated using value_counts(normalize=True) * 100.

- Advanced Assessment - Average Article Length by Sentiment: The average article length is grouped by sentiment category to observe potential variations in length across different sentiment types.

- Most Common Words by Sentiment: The most common words associated with each sentiment category are determined using the Counter class to count the frequency of words. The top 10 most common words are displayed for each sentiment category.

__Pre-cleaning EDA Results and Analysis__
_The pre-cleaning EDA provides the following insights:_

- Dataset Shape: The dataset consists of a certain number of rows and columns, indicating the size of the dataset.

- Missing Values: The number of missing values in each column is shown. This helps identify any data gaps or inconsistencies.

- Unique Sentiment Categories: The unique sentiment categories present in the dataset provide an understanding of the sentiment distribution.

- Sentiment Category Distribution: The distribution of sentiment categories indicates the frequency of each sentiment type, highlighting potential class imbalances.

- Average Article Length: The average length of the news articles in the dataset provides an idea of the article length distribution.

- Data Types: This section reveals the data types of each column, which is important for data preprocessing and modeling.

- Vocabulary Size: The vocabulary size reflects the number of unique words present in the news articles, indicating the diversity of language usage in the dataset.

- Sentiment Category Percentage: The percentage distribution of sentiment categories offers insights into the class distribution, highlighting any potential class imbalances.

- Average Article Length by Sentiment: Analyzing the average article length across different sentiment categories helps identify potential variations in the lengths of articles expressing different sentiments.

- Most Common Words by Sentiment: The most frequently occurring words associated with each sentiment category provide initial information about the language usage and potential sentiment-specific vocabulary.

## Data Cleaning

The dataset used for financial news sentiment analysis requires preprocessing and cleaning before feeding it into the models. This section describes the steps taken to clean the dataset using parallel processing for improved efficiency.

__Cleaning the Text Data:__

The `clean_text_parallel` function is designed to process text data by applying various cleaning techniques. The steps involved in the cleaning process are as follows:

- Lowercasing: The text is converted to lowercase to ensure consistency and avoid duplication of words due to case differences.

- Tokenization: The text is tokenized into individual words to enable further processing on a per-word basis.

- Stop Word Removal: Common English stop words, such as "the," "is," and "and," are removed from the text as they often do not contribute to the sentiment analysis.

- Lemmatization: The words are lemmatized to reduce inflectional forms to their base or dictionary form. This helps to normalize the text and improve the accuracy of sentiment analysis.

- Joining Tokens: The cleaned tokens are joined back together to form the preprocessed text.

- Spelling Correction: Spelling errors are addressed using the TextBlob library, which corrects common spelling mistakes in the text.

- Special Character Removal: Special characters, punctuation, and non-alphabetic characters are removed from the text to eliminate noise and focus on the meaningful words.

- Tokenization after Special Character Removal: The text is tokenized again after special character removal to ensure consistency and prepare for further cleaning steps.

- Removing Numbers: Numeric tokens and individual numbers are removed from the text since they may not contribute significantly to sentiment analysis.

- Joining Tokens: The cleaned tokens are joined back together to form the final cleaned text.

__Parallel Processing:__

To speed up the cleaning process, the `perform_cleaning_parallel` function utilizes parallel processing with multiple processes. The steps involved in parallel processing are as follows:

1. Splitting Dataframe into Chunks: The input dataframe is split into chunks, allowing each process to handle a portion of the data independently.

2. Creating a Multiprocessing Pool: A multiprocessing pool is created with the specified number of processes to distribute the cleaning task.

3. Parallel Cleaning: The clean_text_parallel function is applied in parallel on each chunk of the dataframe, ensuring efficient cleaning of the text data.

4. Concatenating Cleaned Chunks: The cleaned chunks are concatenated back into a single list to obtain the complete cleaned text data.

5. Updating Original Dataframe: The 'news' column in the original dataframe is updated with the cleaned text data.

The use of parallel processing in the cleaning process allows for faster and more efficient text data cleaning, utilizing the available CPU cores and speeding up the preprocessing phase.

Here is the combined function that incorporates both the text cleaning and parallel processing:

``` python
import pandas as pd
import numpy as np
import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from multiprocessing import Pool

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
```

By using this combined function, the text data in the financial news dataset can be effectively cleaned and preprocessed using parallel processing techniques.

## Post-cleaning EDA

After incorporating the cleaning functions, we can compare the results of the post-cleaning EDA with the previous analysis to understand the impact of the cleaning process on the dataset. Here's the updated comparison analysis:

- Dataset Shape: The shape of the dataset remains the same with 4845 rows and 3 columns. The cleaning process did not alter the overall structure of the dataset.

- Missing Values: The dataset still does not have any missing values in the "sentiment," "news," and "article length" columns. The cleaning process did not introduce any new missing values.

- Unique Sentiment Categories: The unique sentiment categories remain the same: 'neutral,' 'negative,' and 'positive.' The cleaning process did not change the sentiment categories present in the dataset.

- Sentiment Category Distribution: The distribution of sentiment categories shows similar proportions with the 'neutral' category being the most frequent, followed by 'positive' and 'negative.' The class imbalance issue is still present even after the cleaning process. Further analysis and modeling techniques may be required to address this class imbalance.

- Average Article Length: The average length of news articles has significantly decreased to approximately 87.98 characters from 128.13 characters in the previous analysis. This decrease in average length can be attributed to the text cleaning process, which involved removing stopwords, special characters, and numbers. The cleaning process has resulted in more concise articles, which may impact the sentiment analysis models' performance.

- Data Types: The data types of the columns remain the same, with 'sentiment' and 'news' being object types, and 'article length' being an integer type. The cleaning process did not affect the data types.

- Vocabulary Size: The vocabulary size has reduced to 7364 from 12971. This reduction indicates that the cleaning process, including the removal of stopwords, special characters, and numbers, has effectively reduced the number of unique words in the dataset. The reduced vocabulary size may simplify the analysis and potentially improve the models' training efficiency.

- Sentiment Category Percentage: The percentage distribution of sentiment categories remains similar, with 'neutral' being the most prevalent, followed by 'positive' and 'negative.' The cleaning process did not significantly alter the distribution of sentiment categories.

- Average Article Length by Sentiment: The average article lengths for each sentiment category have slightly changed but follow the same pattern. The 'positive' sentiment articles still have the longest average length, followed by 'neutral' and 'negative' sentiments. The cleaning process has affected the article lengths but the relative differences between the sentiment categories have been preserved.

- News Length Outliers: The presence of news length outliers has changed, and specific articles have been flagged as outliers based on their length. The cleaning process may have influenced the identification of outliers due to the removal of certain characters and numbers. Outliers can provide valuable insights into unusual or extreme sentiment expressions and may need to be handled appropriately during the modeling process.

- Most Common Words by Sentiment: The most common words for each sentiment category have changed, reflecting the cleaned and preprocessed text. The specific words and their frequencies may vary, but common themes and keywords related to each sentiment category should still be present. The cleaning process has modified the word frequencies, potentially enhancing the sentiment-related information contained in the dataset.

By performing this post-cleaning EDA, we can observe the effects of the cleaning process on various aspects of the dataset. The analysis highlights the changes in article length, vocabulary size, and sentiment category distributions. These insights inform the subsequent steps of model development and help ensure that the cleaned dataset is appropriately prepared for sentiment analysis tasks.

## Splitting the Data

``` python

```

## Building the Model

With our models ready, it's time to train them on our labeled training data. This step involves feeding the models with the input features (X_train) and their corresponding target labels (y_train).

``` python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def build_model():
    # Create the model
    nb_model = MultinomialNB()
    svm_model = SVC()

    return nb_model, svm_model

```

The build_model function is responsible for creating our classification models. We instantiate a Naïve Bayes model (nb_model) and an SVM model (svm_model). These models are widely used in text classification tasks and have shown promising results. We return both models to be trained and evaluated later.

## Training the Model

``` python
def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)

    # Perform model tuning if needed
    # Adjust hyperparameters, such as C or alpha values, based on grid search or cross-validation
    pass

```

The train_model function handles the training process. We simply call the fit method on our chosen model and provide it with the training data. If desired, we can further fine-tune our models by adjusting hyperparameters based on techniques like grid search or cross-validation.

## Evaluating the Model

Once our models are trained, it's crucial to assess their performance on unseen data. We accomplish this by evaluating them using our testing set.

``` python
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, label_encoder):
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Decode the predicted labels
    y_pred = label_encoder.inverse_transform(y_pred)
    y_test = label_encoder.inverse_transform(y_test)

    # Print classification report
    report = classification_report(y_test, y_pred)
    print(report)

```

The evaluate_model function plays a vital role in model evaluation. It takes the trained model, the testing features (X_test), the corresponding labels (y_test), and a label_encoder object used for decoding the labels. The function uses the trained model to make predictions on the test data. It then decodes the predicted and actual labels using the inverse_transform method of the label_encoder. Finally, it generates a classification report, which provides metrics such as precision, recall, and F1-score for each sentiment category.

## Saving the Model

Once we are satisfied with our trained model, it's essential to save it for future use without the need for retraining.

``` python
from sklearn.externals import joblib

def save_model(model, filename):
    # Save the trained model
    joblib.dump(model, filename)
```

The `save_model` function allows us to save our trained model to disk. We utilize the joblib.dump method from Scikit-learn's joblib module. Saving the model ensures that we can use it later for making predictions on new, unseen data without the need to retrain it from scratch.

## Conclusion

To further improve my sentiment analysis models, I have devised the following plan:

- Hyperparameter Tuning: Fine-tune the models by adjusting hyperparameters for optimal performance on sentiment analysis tasks.
- Advanced Feature Representations: Explore alternative feature representations, such as word embeddings or contextualized embeddings, to capture nuanced semantic information and contextual dependencies.
- Data Augmentation: Augment the dataset with synthetic examples or leverage techniques like backtranslation to diversify the training data and enhance generalization capabilities.
- Ensemble Learning: Combine predictions from multiple models to leverage their strengths and improve overall sentiment classification performance.
- Domain-Specific Lexicons: Incorporate industry-specific sentiment lexicons or financial dictionaries to capture domain-specific vocabulary and enhance sentiment classification accuracy in the financial context.

In the mean time there a ton more guides, tutorials and information on my:

- [analysistutorial.com](https://analysistutorial.com/)
- [Tutorial Github Repo](https://github.com/Arash-Nozarinejad/analysis-tutorial)
- [Projects Github Repo](https://github.com/Arash-Nozarinejad/public-projects)
- [Twitter](https://twitter.com/analysistut)
- [Youtube](https://www.youtube.com/@analysistutorial)

You can also connect with me on Linkedin:

- [LinkedIn](https://www.linkedin.com/in/arash-nozarinejad/)
