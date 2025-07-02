# Loan Approval Prediction using Machine Learning
# Author: Sai Karthik (Improved Version)

# =============================
# Step 1: Import required libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # For handling imbalanced data
import joblib

# =============================
# Step 2: Load the dataset
# =============================
# Ensure train.csv.csv is in the same folder as this script
df = pd.read_csv(r"D:\loan\train.csv.csv")

# =============================
# Step 3: Data Exploration
# =============================
# =============================
# Step 3: Data Exploration
# =============================
# =============================
# Step 3: Data Exploration
# =============================
print("🔍 Dataset Info:")
df.info()

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.show()

# Correlation heatmap for numeric features only
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.show()

# Correlation heatmap for numeric features only
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# =============================
# Step 4: Data Preprocessing
# =============================
# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Handle 'Dependents' column
df['Dependents'].replace('3+', 3, inplace=True)
df['Dependents'] = df['Dependents'].astype(int)

# Feature scaling
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# =============================
# Step 5: Handle Imbalanced Data (New)
# =============================
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# =============================
# Step 6: Split into training and testing sets
# =============================
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# =============================
# Step 7: Train the model with Hyperparameter Tuning
# =============================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("🔧 Best Parameters:", grid_search.best_params_)

# =============================
# Step 8: Evaluate the model with Cross-Validation
# =============================
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='f1')
print("📊 Cross-Validation F1-Score:", round(np.mean(cv_scores) * 100, 2), "%")

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("✅ Test Accuracy:", round(accuracy * 100, 2), "%")
print("✅ Precision:", round(precision, 2))
print("✅ Recall:", round(recall, 2))
print("✅ F1-Score:", round(f1, 2))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =============================
# Step 9: Feature Importance
# =============================
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("📈 Feature Importances:\n", feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importances")
plt.show()

# =============================
# Step 10: Predict a sample applicant
# =============================
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = best_model.predict(sample)
print("🔍 Sample Prediction (1 = Approved, 0 = Rejected):", prediction[0])

# =============================
# Step 11: Save the trained model and scaler
# =============================
joblib.dump(best_model, "loan_approval_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model and scaler saved as loan_approval_model.pkl and scaler.pkl")

