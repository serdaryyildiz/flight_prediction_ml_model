**Machine Learning Group Project for CENG 3522 : Applied Machine Learning class with ; <br />
Işıl ÇOLAK : https://github.com/ieamil , <br />
Selman OLGUN : https://github.com/selmanolgun**

# Project Report

### Motivation
*Flying is an essential mode of transportation for millions of people worldwide. However, flight cancellations disrupt travel plans, causing inconvenience and frustration for passengers. Identifying the factors that lead to flight cancellations and predicting them in advance can greatly benefit both airlines and passengers. Our goal is to develop a model that accurately predicts the likelihood of flight cancellations. *

### Methods
Data Collection 
Determining Data Sources: For flight data, we utilized the US Department of Transportation, Bureau of Transportation Statistics. We pulled the weather data we considered adding to the flight data from https://www.wunderground.com/history using the Python Selenium library. 

### Data Processing
Data Filtering: We collected nearly 55,000 weather data entries, which took approximately 3 days. 
Dropping Irrelevant Columns: Initially, our dataset comprised 32 columns, but not all were relevant to our analysis. We removed columns unrelated to our project. 
Handling Missing Values: Our dataset was large, and we aimed to reduce its size. To achieve this, we removed rows with null values. This not only decreased file size but also enhanced our ability to manage missing data effectively. 
Scraping Weather Data: Our dataset included information about departure city, arrival city, and flight date. To analyze the impact of weather on flight cancellations, we used this data to scrape weather information for each flight from the internet using the Python Selenium library. We focused on specific weather details (e.g., temperature, wind) and integrated this information into our dataset. We created separate new columns for both departure city and arrival city information for each flight record, as we wanted data for both cities. We merged the collected data and removed rows with null values. One-hot encoding was applied to the "CONDITION_ORIGIN" and "CONDITION_DEST" columns.

### Machine Learning Algorithm
We implemented a neural network algorithm using the MLPClassifier with Scikit-learn. Since Scikit-learn's MLPClassifier does not directly support Dropout, we couldn't utilize Dropout. Instead, we attempted to mitigate overfitting by reducing the number of hidden layers and adding class weights. We set the number of neurons in the hidden layers to 16 and 8, respectively, while the output layer consisted of 1 neuron. We employed the Relu activation function. 

### Performance Evaluation
We increased the weight of the weather condition columns. To prevent overfitting, we utilized early stopping. Additionally, we applied L2 Regularization. Early stopping and validation accuracy were improved by adding the validation_fraction parameter. We implemented early stopping using the n_iter_no_change parameter.

### Results

Test Accuracy: 0.9947798851574735
Test Loss: 0.019454094406459867

		Classification Report :  
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     11269
         1.0       0.88      0.84      0.86       225

    accuracy                           0.99     11494
   macro avg       0.94      0.92      0.93     11494
weighted avg       0.99      0.99      0.99     11494

Due to the test accuracy results, we suspected overfitting in the model. Subsequently, we decided to add validation tests. The results of the validation tests are as follows:

Validation Accuracy: 0.9944535073409462
Validation Loss: 0.02770154964712071

		Classification Report for Validation Set:  
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      9022
         1.0       0.88      0.82      0.85       173

    accuracy                           0.99      9195
   macro avg       0.94      0.91      0.92      9195
weighted avg       0.99      0.99      0.99      9195

### Conclusion
 The model performed well in validation tests, providing promising results. However, the predictive ability of the model could be further enhanced by using more data. We find the study promising, as it can be developed into a feasible and applicable real-world project with additional data and work.


###References
•	https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?select=flights_sample_3m.csv
•	https://www.wunderground.com/history
•	https://www.geeksforgeeks.org/classification-using-sklearn-multi-layer-perceptron/
