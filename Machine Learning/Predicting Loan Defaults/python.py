
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


def load_dataset(file_address: str) -> pd.DataFrame:
    '''
    Loads the dataset using pandas.read_csv()
    '''
    dataset = pd.read_csv(file_address)

    return dataset


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


def split_train_test_data(dataset: pd.DataFrame, target_column: str, test_size=0.2, random_state=42) -> tuple:
    '''
    Using sklearn's train_test_split function, we'll split the data into train and test split
    '''

    x = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    print('Tran set shape: ', x_train.shape, y_train.shape)
    print('Test set shape: ', x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test


def train_evaluate_logistic_regression(x_train, y_train, x_test, y_test):
    '''
    This is a simple Sklearn logistic regression training and evaluation function.
    Evaluation is done using:

        1. Accuracy
        2. Precision
        3. Recall
        4. F1 Score
    '''

    # Model training
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Model evaluation
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print('Logistic Regression Model Evaluation: ')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)


def train_evaluate_decision_tree(x_train, y_train, x_test, y_test):
    '''
    This is a decision tree training and evaluation function.
    Evaluation is done using:

        1. Accuracy
        2. Precision
        3. Recall
        4. F1 Score
    '''
    
    # Model training
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # Model evaluation
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
        
    # Print evaluation metrics
    print("Decision Tree Model Evaluation:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def train_evaluate_random_forest(x_train, y_train, x_test, y_test):
    '''
    This is an advanced ensemble evaluation algorithm called random forest.
    Evaluation is done using:

        1. Accuracy
        2. Precision
        3. Recall
        4. F1 Score
    '''

    # Model training
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Model evaluation
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
        
    # Print evaluation metrics
    print("Decision Tree Model Evaluation:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def train_evaluate_models(dataset: pd.DataFrame, target_column: str, test_size=0.2, random_state=42) -> None:
    '''
    This function first splits the dataset into train and test data. Then, Three models are used to 
    train and evaluate the data. These models are:
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
    '''

    x_train, x_test, y_train, y_test = split_train_test_data(dataset, target_column, test_size, random_state)

    train_evaluate_logistic_regression(x_train, y_train, x_test, y_test)

    train_evaluate_decision_tree(x_train, y_train, x_test, y_test)

    train_evaluate_random_forest(x_train, y_train, x_test, y_test)


def run(file_address: str):
    '''
    '''
    dataset = load_dataset(file_address)

    dataset = clean_dataset(dataset)

    # perform_eda(dataset)

    dataset = preprocess_data(dataset)

    dataset = dataset[feature_engineering_selection(dataset)]

    train_evaluate_models(dataset, 'Status')



if __name__ == '__main__':
    file_address = 'dataset.csv'
    run(file_address)
