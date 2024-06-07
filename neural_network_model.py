import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("./balanced_data_final.csv")
data = pd.DataFrame(df)

X_categories = data.drop(["FL_DATE", "DOT_CODE", "FL_NUMBER"], axis=1)
x = pd.get_dummies(X_categories, drop_first=True)
y = data["CANCELLED"]

important_features = x.iloc[:, 26:]
x = pd.concat([x, important_features, important_features], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

mlp = MLPClassifier(hidden_layer_sizes=(16,8), activation='relu', solver='adam', max_iter=300,alpha=0.001 , random_state=42, early_stopping=True,
                    n_iter_no_change=10 , validation_fraction=0.1)
# Alpha = 0.001 = L2 Regularization

history = mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, mlp.predict_proba(X_test))

y_pred_val = mlp.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
val_loss = log_loss(y_val, mlp.predict_proba(X_val))

training_loss = history.loss_curve_
training_accuracy = history.validation_scores_

epochs = range(1, len(training_accuracy) + 1)

# Accuracy Plot
plt.plot(epochs, training_accuracy, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Plot
plt.plot(epochs, training_loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

classification_rep = classification_report(y_test, y_pred)
classification_rep_val = classification_report(y_val, y_pred_val)

confusion_matrix_validation = confusion_matrix(y_val, y_pred_val)
confusion_matrix_test = confusion_matrix(y_test, y_pred)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss}')
print("\n\t\tClassification Report :  \n", classification_rep)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Loss: {val_loss}')
print("\n\t\tClassification Report for Validation Set:  \n", classification_rep_val)

print(f'Confusion Matrix (Validation Set) :\n {confusion_matrix_validation}\n\n')
print(f'Confusion Matrix (Test Set) :\n {confusion_matrix_test}')

print(data["CANCELLED"].value_counts())

# Confusion Matrix for Validation Set
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_validation, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for Test Set
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
