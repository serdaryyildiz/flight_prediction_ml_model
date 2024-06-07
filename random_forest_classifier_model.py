import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Data
df = pd.read_csv("balanced_data_final.csv")
# Drop useless columns
X_categories = df.drop(["FL_DATE", "DOT_CODE", "FL_NUMBER"], axis=1)
x = pd.get_dummies(X_categories, drop_first=True)
y = df["CANCELLED"]

# Standardizing data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Reduces the dimensionality of the dataset to 50 principal components.
pca = PCA(n_components=50)
x_pca = pca.fit_transform(x_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=1)

#We are using Random Forest Classifier (set of decision trees)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) # Cross - Validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

print(f'\nCross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}\n')

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, rf.predict_proba(X_test))

#Output Reports
classification_rep = classification_report(y_test, y_pred)
confusion_matrix_test = confusion_matrix(y_test, y_pred)

# Display results
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss}')
print("\n\t\tClassification Report :  \n", classification_rep)
print(f'Confusion Matrix (Test Set) :\n {confusion_matrix_test}')

# Confusion Matrix Graph for Test Set
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

epochs = range(1, len(cv_scores) + 1)
# Cross-validation acuracy plot
plt.plot(epochs, cv_scores, 'b', label='Cross-validation accuracy')
plt.title('Cross-validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(df["CANCELLED"].value_counts())
