import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load the data
df = pd.read_csv("loan_data.csv")

# Create new features
df['log_income'] = np.log(df['income'])
df['log_loan_amount'] = np.log(df['loan_amount'])
df['term_binary'] = df['term'].apply(lambda x: 1 if x > 12 else 0)

# Drop rows with any missing values (NaNs)
df.dropna(inplace=True)

# Define features and target
features = ['log_income', 'log_loan_amount', 'credit_history', 'term_binary']
target = 'defaulted'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
