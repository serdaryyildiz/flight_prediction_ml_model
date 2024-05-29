import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("C:/Users/serda/data_with_forecast.csv")
data = pd.DataFrame(df)

X_categories = data.drop(["FL_DATE", "DOT_CODE", "FL_NUMBER"], axis=1)
x = pd.get_dummies(X_categories, drop_first=True)
y = data["CANCELLED"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.01, batch_size=32,
                    learning_rate_init=0.001, max_iter=50, random_state=1, early_stopping=True, n_iter_no_change=3)

history = mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, mlp.predict_proba(X_test))

training_loss = history.loss_curve_
training_accuracy = history.validation_scores_

epochs = range(1, len(training_accuracy) + 1)

#Accuracy Plot
plt.plot(epochs, training_accuracy, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Loss Plot
plt.plot(epochs, training_loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss}')
