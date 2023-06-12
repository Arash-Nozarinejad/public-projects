# Predicting Loan Defaults

__Introduction:__

Welcome to the Predicting Loan Default project! This project aims to predict whether a loan will default based on various features. It's a wonderful opportunity to learn how to build a complete machine learning project from scratch.

__Table of Content:__

- [Predicting Loan Defaults](#predicting-loan-defaults)
  - [1.Understanding the Problem](#1understanding-the-problem)
  - [2.Data Collection](#2data-collection)
  - [3.Data Cleaning](#3data-cleaning)
  - [3.5 Data Imputing](#35-data-imputing)
  - [4.Data Preprocessing](#4data-preprocessing)
  - [5.Exploratory Data Analysis](#5exploratory-data-analysis)
  - [6.Feature Engineering and Selection](#6feature-engineering-and-selection)
  - [7.Model Training \& Model Evaluation](#7model-training--model-evaluation)
  - [8. Results](#8-results)
  - [conclusion](#conclusion)

__Python libraries used:__

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
```

---

## 1.Understanding the Problem

The initial step in this project is to gain a clear understanding of the problem we are addressing. The objective is to predict whether a loan will default by leveraging a range of loan-related features.

A loan default refers to a situation where a borrower fails to fulfill their financial obligation by not repaying the borrowed amount in accordance with the agreed-upon terms and conditions. It is crucial to comprehend the concept of loan default to grasp the significance of accurately predicting and managing such occurrences.

---

## 2.Data Collection

The next step is to collect the necessary data for our analysis and modeling. In this project, we obtained the loan default dataset from Kaggle, a popular online platform for datasets and machine learning resources. The dataset can be accessed and downloaded from the following Kaggle link: [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset).

To streamline the data import process, we created a Python function called "load_dataset" specifically designed to load the dataset into our Python environment. This function allows us to easily retrieve and access the dataset for subsequent data preprocessing and analysis tasks.

``` python
def load_dataset(file_address: str) -> pd.DataFrame:
    '''
    Loads the dataset using pandas.read_csv()
    '''
    dataset = pd.read_csv(file_address)

    return dataset
```

---

## 3.Data Cleaning

In this phase, our focus is on ensuring that the dataset is clean and suitable for further analysis. We perform the following steps to achieve this:

1. __Removing Unnecessary Columns__: We identify and eliminate any columns that are not relevant to our analysis. By removing these columns, we reduce the dimensionality of the dataset and improve computational efficiency.

2. __Converting Data Types__: We address any inconsistencies in data types to ensure that each column has the appropriate data type. This step is crucial for accurate calculations and modeling. For example, we may convert numerical variables from strings to float or integer data types.

3. __Cleaning Categorical Variables__: Categorical variables require special attention as they often contain textual or categorical representations. We apply cleaning techniques to standardize the values, remove leading/trailing spaces, and make them consistent for analysis and modeling.

However, a challenge arises when dealing with rows that have missing values. Simply removing these rows may result in a significant loss of important data, hampering the completeness of our analysis. Therefore, we need to perform imputation, which involves estimating or filling in the missing values with appropriate substitutes. Imputation allows us to retain valuable information and ensures a more comprehensive analysis.

## 3.5 Data Imputing

``` python
def impute_missing_values(old_dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    This function imputes missing values of the dataset by:

        1. Identifying columns with missing values
        2. Imputing numerical values
        3. Imputing categorical columns
    '''
    
    dataset = old_dataset

    # Identify columns with missing values
    columns_with_missing = dataset.columns[dataset.isnull().any()].tolist()
    
    # Separate numerical and categorical columns
    numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = dataset.select_dtypes(include=['object']).columns

    # Impute numerical columns with mean
    numerical_imputer = SimpleImputer(strategy='mean')
    dataset[numerical_columns] = numerical_imputer.fit_transform(dataset[numerical_columns])

    # Impute categorical columns with mode
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    dataset[categorical_columns] = categorical_imputer.fit_transform(dataset[categorical_columns])

    return dataset
```

And now we can write the _clean\_dataset_ function.

``` python
def clean_dataset(old_dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    This function cleans the dataset by:

        1. Imputing missing values
        2. Removing Unnecessary columns
        3. Converts data types
        4. Cleans categorical variables
    '''

    dataset = old_dataset

    # Impute missing values
    dataset = impute_missing_values(dataset)

    # Remove unnecessary columns
    dataset = dataset.drop(columns=['ID'])

    # Reset the indices
    dataset.reset_index(drop=True, inplace=True)

    # Convert data types
    numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    dataset[numerical_columns] = dataset[numerical_columns].astype(float)

    categorical_columns = dataset.select_dtypes(include=['object']).columns
    dataset[categorical_columns] = dataset[categorical_columns].astype(str)
   

    # Clean categorical variables
    dataset['Region'] = dataset['Region'].str.lower().str.strip()
    dataset['Gender'] = dataset['Gender'].str.lower().str.strip()
    dataset['loan_type'] = dataset['loan_type'].str.lower().str.strip()
    dataset['Secured_by'] = dataset['Secured_by'].str.lower().str.strip()
    dataset['loan_limit'] = dataset['loan_limit'].str.lower().str.strip()
    dataset['credit_type'] = dataset['credit_type'].str.lower().str.strip()
    dataset['total_units'] = dataset['total_units'].str.lower().str.strip()
    dataset['loan_purpose'] = dataset['loan_purpose'].str.lower().str.strip()
    dataset['interest_only'] = dataset['interest_only'].str.lower().str.strip()
    dataset['approv_in_adv'] = dataset['approv_in_adv'].str.lower().str.strip()
    dataset['Security_Type'] = dataset['Security_Type'].str.lower().str.strip()
    dataset['occupancy_type'] = dataset['occupancy_type'].str.lower().str.strip()
    dataset['lump_sum_payment'] = dataset['lump_sum_payment'].str.lower().str.strip()
    dataset['Credit_Worthiness'] = dataset['Credit_Worthiness'].str.lower().str.strip()
    dataset['Neg_ammortization'] = dataset['Neg_ammortization'].str.lower().str.strip()
    dataset['construction_type'] = dataset['construction_type'].str.lower().str.strip()
    dataset['business_or_commercial'] = dataset['business_or_commercial'].str.lower().str.strip()
    dataset['co-applicant_credit_type'] = dataset['co-applicant_credit_type'].str.lower().str.strip()
    dataset['submission_of_application'] = dataset['submission_of_application'].str.lower().str.strip()

    return dataset
```

## 4.Data Preprocessing

Once the data cleaning step is complete, we move on to data preprocessing. This stage involves transforming the data in preparation for model training and analysis. In our case, we perform two key preprocessing steps:

1. __Encoding Categorical Variables__: Categorical variables are typically represented as text or labels that are not suitable for most machine learning algorithms. To address this, we employ an encoding technique to convert categorical variables into a numerical representation. This ensures compatibility with various algorithms. Common encoding methods include label encoding and one-hot encoding.

2. __Scaling Numerical Variables__: Numerical variables often have varying scales, which can adversely affect the performance of certain machine learning algorithms. To address this issue, we apply scaling techniques to normalize the numerical variables. Scaling ensures that each feature contributes equally to the analysis and prevents any single feature from dominating the model. Common scaling methods include standardization (using the mean and standard deviation) and normalization (scaling to a specific range).

By performing these preprocessing steps, we enhance the quality and compatibility of the data for subsequent model training and analysis.

``` python
def preprocess_data(old_dataset: pd.DataFrame) -> pd.DataFrame:
    '''
     This function proprocesses the dataset by:

        1. Encoding categorical variables
        2. Sclading numerical variables
    '''

    dataset = old_dataset

    # Encode categorical variables
    categorical_columns = ['year', 'loan_limit', 'Gender', 'approv_in_adv', 'loan_type',
                           'loan_purpose', 'Credit_Worthiness', 'open_credit', 'business_or_commercial',
                           'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'construction_type',
                           'occupancy_type', 'Secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type',
                           'age', 'submission_of_application', 'Region', 'Security_Type', 'Status']
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        dataset[column] = label_encoder.fit_transform(dataset[column])

    # Scale numerical variables
    numerical_columns = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term',
                         'income', 'Credit_Score', 'LTV', 'dtir1']
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

    # Return the preprocessed data
    return dataset
```

---

## 5.Exploratory Data Analysis

In this phase, we delve deeper into our dataset to gain insights and a better understanding of the relationships between variables. EDA involves various techniques to explore and visualize the data. We perform the following key steps:

1. __Visualizing the Distribution of the Target Variable__: Understanding the distribution of the target variable is crucial as it provides insights into the class balance and informs us about the prevalence of loan defaults. By visualizing the distribution, such as through a bar plot or histogram, we can assess the class proportions and identify any class imbalance issues.

2. __Visualizing the Correlation Matrix__: The correlation matrix depicts the relationships between numerical variables. By visualizing the correlation matrix using techniques like a heatmap, we can identify strong positive or negative correlations between variables. This helps us understand which features are highly correlated and can guide feature selection or uncover potential multicollinearity issues.

3. __Exploring Categorical Variables__: Categorical variables offer valuable insights into different groups or categories within our data. We can explore these variables by creating frequency tables or bar plots to visualize the distribution of each category. This analysis helps us identify patterns, imbalances, or potential relationships between categorical variables and the target variable.

4. __Exploring Numerical Variables__: Numerical variables contain continuous or discrete numerical values. We can examine these variables through descriptive statistics such as mean, median, standard deviation, and quartiles. Additionally, we can create histograms or box plots to visualize the distribution and identify any outliers or skewed distributions.

By performing EDA, we gain a deeper understanding of the dataset, identify potential issues or patterns, and make informed decisions regarding data preprocessing, feature selection, and model building.

``` python
def perform_eda(dataset: pd.DataFrame) -> None:
    '''
    This function performs Exploratory Data Analysis by:

        1. Visualizing the distribution of the target variable
        2. Visualizing the correlation matrix
        3. Exploring categorical variables
        4. Exploring numerical Variables
    '''

    # Visualize the distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataset, x="Status")
    plt.title('Distribution of Loan Status')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.show()

    # Visualize the correlation matrix
    plt.figure(figsize=(20, 12))
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='RdBu', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Explore categorical variables
    categorical_columns = ['year', 'loan_limit', 'Gender', 'approv_in_adv', 'loan_type',
                           'loan_purpose', 'Credit_Worthiness', 'open_credit', 'business_or_commercial',
                           'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'construction_type',
                           'occupancy_type', 'Secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type',
                           'age', 'submission_of_application', 'Region', 'Security_Type', 'Status']
    for column in categorical_columns:
        plt.figure(figsize=(10, 8))
        sns.countplot(x=column, data=dataset, hue='Status')
        plt.title(f'Distribution of Load Status by {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Status', loc='upper right')
        plt.show()


    # Explore numerical variables
    numerical_columns = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term',
                         'income', 'Credit_Score', 'LTV', 'dtir1']
    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxenplot(x='Status', y=column, data=dataset)
        plt.title(f'Distribution of {column} by Loan Status')
        plt.xlabel('Status')
        plt.ylabel(column)
        plt.show()
```

---

## 6.Feature Engineering and Selection

In this step, we focus on enhancing the predictive power of our dataset by creating new features or modifying existing ones. Feature engineering involves leveraging domain knowledge and insights gained from the EDA phase to extract relevant information from the existing variables. The goal is to create features that capture important patterns or relationships, potentially improving the performance of our models.

To begin, we implement a function that adds two new features. These features are designed to provide additional insights into the data. For example, we may calculate ratios, differences, or other derived values based on the existing variables. These new features can potentially capture important relationships or patterns that were not evident in the original dataset.

After creating the new features, we perform feature elimination to further refine the feature set and improve model performance. This involves selecting a subset of the most relevant features while discarding those that may not contribute significantly to the prediction task. Two common feature elimination algorithms employed in this step are:

1. __Backward Elimination using Ordinary Least Squares__: This algorithm involves iteratively fitting a model using ordinary least squares (OLS) regression and eliminating features based on their p-values. The algorithm starts with a full model containing all features and then removes one feature at a time based on statistical significance until a desired level of significance is achieved.

2. __Recursive Feature Elimination using Linear Regression__: In this algorithm, we fit a linear regression model and recursively eliminate features based on their importance or contribution to the model. The algorithm ranks the features by their relevance and removes the least important features iteratively until the desired number of features is reached.

By applying these feature engineering and elimination techniques, we aim to improve the performance and interpretability of our models by focusing on the most relevant and informative features for loan default prediction.

``` python
def feature_engineering_selection(dataframe: pd.DataFrame):
    '''
    This function adds two new features first. Then, performs feature elimination
    to improve model performance. Feature elimination algorithms:

        1. Backward Elimination using 'ordinary least squares'
        2. Recursive Feature Elimination using 'linear regression'
    
    '''
    dataset = dataframe

    # Ratio of loan_amount to property_value
    dataframe['loan_to_value_ratio'] = dataframe['loan_amount'] / dataframe['property_value']
    
    # Total credit available
    dataframe['total_credit_available'] = dataframe['loan_limit'] - dataframe['loan_amount']

    # Ratio of loan to income ratio
    dataframe['loan_to_income_ratio'] = dataframe['loan_amount'] / dataframe['income']

    # Difference between loan amount and upfront charges
    dataframe['loan_amount_diff_upfront_charges'] = dataframe['loan_amount'] - dataframe['Upfront_charges']

    # Perform feature selection using backward elimination
    x = dataset.drop(columns=['Status'])
    y= dataset['Status']
    x = sm.add_constant(x)

    # Fit the ordinary least squares (OLS) model
    model = sm.OLS(y, x).fit()

    # Perform backward elimination
    while len(x.columns) > 1:
        p_values = model.pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value > 0.05:
            max_p_value_column = p_values.idmax()
            x = x.drop(columns=max_p_value_column)
            model = sm.OLS(x, y).fit()
        else:
            break
    
    # Perform feature selection using Recursive Feature Elimination (RFE)
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector.fit(x, y)

    # Print the selected_features
    selected_features = x.columns[selector.support_]
    selected_features = selected_features.tolist() + ['Status']
    print('Selected features: ', selected_features)

    # Print the summary of the final model
    print(model.summary())

    return selected_features
```

---

## 7.Model Training & Model Evaluation

In this crucial stage, we train and evaluate predictive models using our prepared dataset. We begin by splitting the dataset into training and testing subsets to assess the performance of the models on unseen data. The primary train/evaluate function carries out this step.

We then proceed to train and evaluate three different models:

1. __Logistic Regression__: Logistic regression is a popular classification algorithm that models the relationship between the input features and the target variable using a logistic function. It is a widely used method for binary classification tasks like loan default prediction.

2. __Decision Tree__: Decision trees are versatile algorithms that make predictions by creating a tree-like model of decisions and their possible consequences. They are capable of capturing complex interactions between variables and are well-suited for classification tasks.

3. __Random Forest__: Random forest is an advanced ensemble learning algorithm that combines multiple decision trees to make predictions. It is known for its ability to reduce overfitting, handle high-dimensional datasets, and improve prediction accuracy. Random forest aggregates the predictions of multiple decision trees to make the final prediction.

For the evaluation of these models, we utilize several common evaluation metrics:

1. __Accuracy__: Accuracy measures the overall correctness of the model's predictions by comparing them to the actual target values.

2. __Precision__: Precision quantifies the proportion of correctly predicted positive instances (loan defaults) out of all instances predicted as positive. It assesses the model's ability to minimize false positives.

3. __Recall__: Recall, also known as sensitivity or true positive rate, measures the proportion of correctly predicted positive instances out of all actual positive instances. It evaluates the model's ability to capture all positive instances and minimize false negatives.

4. __F1 Score__: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance by considering both precision and recall simultaneously.

By training and evaluating these models using various metrics, we can assess their performance and choose the most suitable model for loan default prediction based on their respective strengths and weaknesses.

``` python

```

---

## 8. Results

After training and evaluating the three models (Logistic Regression, Decision Tree, Random Forest) on our loan default prediction task, we obtained the following results:

Logistic Regression Model Evaluation:

- Accuracy: 0.7696
- Precision: 0.7997
- Recall: 0.0717
- F1 Score: 0.1316

Decision Tree Model Evaluation:

- Accuracy: 0.9768
- Precision: 0.9217
- Recall: 0.9887
- F1 Score: 0.9540

Random Forest Model Evaluation:

- Accuracy: 0.9775
- Precision: 0.9234
- Recall: 0.9895
- F1 Score: 0.9553

Upon analyzing these results, we can draw the following observations and make reasonable conclusions:

1. __Accuracy__: The accuracy metric represents the overall correctness of the model's predictions. In our case, both the Decision Tree and Random Forest models achieved high accuracy scores (around 97.7%), indicating that they are able to predict loan defaults with a high degree of correctness. The Logistic Regression model had a lower accuracy of 76.9%.

2. __Precision__: Precision measures the model's ability to correctly identify loan defaults out of all instances predicted as defaults. Both the Decision Tree and Random Forest models achieved high precision scores (around 92.3% and 92.4%, respectively), indicating a low number of false positives. The Logistic Regression model also had a decent precision score of 79.9%.

3. __Recall__: Recall quantifies the model's ability to capture all actual positive instances (loan defaults) correctly. The Decision Tree and Random Forest models achieved high recall scores (around 98.9%), suggesting that they can effectively identify the majority of loan defaults. The Logistic Regression model had a lower recall score of 7.2%, indicating that it may have difficulty capturing loan defaults.

4. __F1 Score__: The F1 score is a balanced measure that combines precision and recall. It provides an overall evaluation of the model's performance by considering both metrics simultaneously. Both the Decision Tree and Random Forest models achieved high F1 scores (around 95.5% and 95.3%, respectively), reflecting a good balance between precision and recall. The Logistic Regression model had a lower F1 score of 13.2%.

Based on these results, it can be concluded that the Decision Tree and Random Forest models outperform the Logistic Regression model in terms of accuracy, precision, recall, and F1 score. These ensemble models demonstrate superior performance in capturing loan defaults, thanks to their ability to handle complex relationships and capture interactions between variables.

## conclusion

My plan is to further improve the project later by adding:

- Experiment with different machine learning models and try deep learning
- Find faulty data to explore more sophisticated methods of handling missing data and outliers
- Conduct a more thorough feature engineering and selection process
- Deploy the model as a real-wold API and test its performance
- Develop a user-interface to take user inputs and return predictions.

In the mean time there a ton more guides, tutorials and information on my:

- [analysistutorial.com](https://analysistutorial.com/)
- [Tutorial Github Repo](https://github.com/Arash-Nozarinejad/analysis-tutorial)
- [Projects Github Repo](https://github.com/Arash-Nozarinejad/public-projects)
- [Twitter](https://twitter.com/analysistut)
- [Youtube](https://www.youtube.com/@analysistutorial)

You can also connect with me on Linkedin:

- [LinkedIn](https://www.linkedin.com/in/arash-nozarinejad/)
