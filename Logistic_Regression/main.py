#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

import pandas as pd

#%%

# import cars.csv
df = pd.read_csv('../Data/cars.csv')

#%%

# Create a OneHotEncoder instance
encoder = OneHotEncoder()

# Fit the encoder to the 'origin' column and transform it
encoded_origin = encoder.fit_transform(df[['origin']])

# Convert the encoded origin column into a DataFrame
encoded_origin_df = pd.DataFrame(encoded_origin.toarray(), columns=encoder.get_feature_names_out(['origin']))

# Concatenate the original DataFrame 'df' with the encoded origin DataFrame
df = pd.concat([df, encoded_origin_df], axis=1)

# Drop the original 'origin' column
df.drop(columns=['origin'], inplace=True)

#%%

X = df[['weight']]

# Assuming 'origin_USA' is the one-hot encoded column for USA
y = df['origin_1']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#%%

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

#%%

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Model accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# %%

# Save the model
dump(model, 'logistic_regression_model.joblib')

# %%

# Create a Logistic Regression model with different hyperparameters
model = LogisticRegression(C=0.5, penalty='l2', solver='saga')

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate and print the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

#%%

# Save the model
dump(model, 'logistic_regression_model_tuned.joblib')

#%%

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their values
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create a Logistic Regression model
model = LogisticRegression()

# Create the Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions on the test data using the best model
y_pred = grid_search.predict(X_test)

# Calculate and print the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

#%%

# Save the model
dump(grid_search.best_estimator_, 'logistic_regression_model_grid_search.joblib')

#%%

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(cm, columns=['Predicted Positive', 'Predicted Negative'], 
                     index=['Actual Positive', 'Actual Negative'])

plt.figure(figsize=(10,7))
cax = plt.matshow(cm, cmap='Blues')
plt.colorbar(cax)

# Add labels to the plot
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.title('Confusion Matrix')
plt.xticks([0, 1], ['Positive', 'Negative'])
plt.yticks([0, 1], ['Positive', 'Negative'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
