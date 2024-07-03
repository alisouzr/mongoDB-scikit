from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Create the database and collection
db = client['my_data_base']
collection = db['my_collection']

# Insert sample data
sample_data = [
    {'feature1': 5.1, 'feature2': 3.5, 'feature3': 1.4, 'feature4': 0.2, 'target': 0},
    {'feature1': 4.9, 'feature2': 3.0, 'feature3': 1.4, 'feature4': 0.2, 'target': 0},
    {'feature1': 6.2, 'feature2': 3.4, 'feature3': 5.4, 'feature4': 2.3, 'target': 1},
    {'feature1': 5.9, 'feature2': 3.0, 'feature3': 5.1, 'feature4': 1.8, 'target': 1},
    # Add more data as needed
]

collection.insert_many(sample_data)
print("Data inserted successfully!")

# Load data from the collection
data = list(collection.find())

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Assuming the data has a 'target' column we want to predict and the others are features
X = df.drop(columns=['target', '_id'])  # '_id' is an automatic MongoDB column that we need to remove
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)