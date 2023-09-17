# Oysis-Infobyte-Task1
 
# Iris Species Classification

Iris flower has three species; setosa, versicolor, and virginica, which differs according to their
measurements. Now assume that you have the measurements of the iris flowers according to
their species, and here your task is to train a machine learning model that can learn from the
measurements of the iris species and classify them.

This is a simple Python script that uses Logistic Regression to classify the species of Iris flowers based on their sepal length, sepal width, petal length, and petal width.

## Libraries Used
- pandas
- numpy
- sklearn

## Data
The data used in this script is the Iris dataset. It includes the following columns:
- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

The 'Species' column is the target variable which we want to predict.

## How to Run the Script
1. Import the necessary libraries.
2. Load the Iris dataset.
3. Define the features (X) and the target variable (y).
4. Split the data into training and testing sets using `train_test_split` from sklearn.
5. Create a Logistic Regression model.
6. Train the model using the training data.
7. Make predictions on the testing data.
8. Evaluate the model by calculating the accuracy of the predictions.

## Output
The script prints the accuracy of the model in percentage format.

## Future Improvements
This is a simple script and there's room for improvement. Here are a few things you can do:
- Perform exploratory data analysis (EDA) to understand the data better.
- Try different machine learning models and compare their performance.
- Optimize the model by tuning hyperparameters.

