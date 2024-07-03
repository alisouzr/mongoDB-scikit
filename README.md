## Project Description: MongoDB Data Analysis with Scikit-learn

### Overview
This project demonstrates how to perform data analysis and machine learning using a non-relational (NoSQL) database, specifically MongoDB, in combination with scikit-learn, a popular machine learning library in Python. The project covers the following steps:
1. Setting up and connecting to MongoDB. (bd.py)
2. Creating a database and a collection.
3. Inserting sample data into the collection.
4. Loading the data from MongoDB into a pandas DataFrame.
5. Preprocessing the data.
6. Training a logistic regression model.
7. Evaluating the model's performance.

### Prerequisites
- MongoDB installed and running.
- Python installed with the following packages: `pymongo`, `pandas`, `scikit-learn`.
- or just copy this command `pip install -r requirements.txt`

### Steps

#### 1. Setup and Connect to MongoDB
First, we set up a connection to the MongoDB server. This can be a local instance or a remote server, and it can include authentication if necessary.

#### 2. Create Database and Collection
We create a new database and a collection within MongoDB to store our sample data. In this example, the database is named `my_data_base` and the collection is `my_collection`.

#### 3. Insert Sample Data
We insert a set of sample data into the collection. This data includes features and a target variable that we aim to predict.

#### 4. Load Data into a pandas DataFrame
We retrieve the data from MongoDB and load it into a pandas DataFrame. This step facilitates easy manipulation and preprocessing of the data.

#### 5. Preprocess the Data
We preprocess the data by separating the features from the target variable and splitting the dataset into training and test sets. We also standardize the features to have zero mean and unit variance using `StandardScaler`.

#### 6. Train a Logistic Regression Model
Using scikit-learn, we train a logistic regression model on the training data.

#### 7. Evaluate the Model
We make predictions on the test data and evaluate the model's performance using metrics such as accuracy and a classification report.

### Code

You can se the code in **mongodb-scikit.py**

### Explanation

1. **MongoDB Connection**: We use `MongoClient` to connect to the MongoDB server.
2. **Database and Collection Creation**: We create a database named `my_data_base` and a collection named `my_collection`.
3. **Data Insertion**: We insert a list of dictionaries representing sample data into the collection.
4. **Data Loading**: We retrieve the data from MongoDB and convert it to a pandas DataFrame.
5. **Data Preprocessing**: We separate features and target variable, split the data into training and test sets, and standardize the features.
6. **Model Training**: We train a logistic regression model using the training data.
7. **Model Evaluation**: We evaluate the model's performance on the test data using accuracy and classification report metrics.

This project provides a comprehensive workflow for integrating MongoDB with scikit-learn for data analysis and machine learning tasks.