# Elevate-labs
AIML internship tasks and projects


TASK 1

1.Titanic dataset analysis using Python and Jupyter Notebook

2.Handled missing values in Age, Embarked, and Cabin columns

3.Applied Label Encoding on Sex, Embarked, and derived Deck feature

4.Normalized numerical features: Age, Fare, SibSp, and Parch

5.Visualized outliers using seaborn boxplots

6.Used pandas, numpy, matplotlib, seaborn, and scikit-learn libraries

7.Dataset sourced from Kaggle Titanic competition

8.Cleaned and transformed data ready for machine learning

9.Includes data preprocessing, encoding, and scaling in one pipeline

10.All analysis is done inside analysis.ipynb notebook

TASK 2

1.Loaded the Titanic Dataset using Pandas for analysis.

2.Generated Summary Statistics (mean, median, standard deviation, etc.) to understand distributions of numerical features.

3.Plotted Histograms for each numeric column to visualize the spread and shape of data distributions.

4.Created Boxplots to detect the presence of outliers in numerical features.

5.Calculated the Correlation Matrix to study how features are related to each other numerically.

6.Visualized the Correlation Matrix using a Seaborn heatmap for better pattern identification.

7.Plotted a Pairplot of selected important features (Survived, Pclass, Sex, Age, Fare, SibSp, Parch) to explore feature relationships grouped by survival status.

8.Built an Interactive Scatter Plot (using Plotly) to explore relationships between Age and Fare colored by the survival outcome.

TASK 3

1.  Import Libraries: Imports necessary libraries including pandas for data manipulation, matplotlib for plotting, scikit-learn for model building and evaluation.
2.  Load Data: Reads the housing data from the `Housing.csv'  file into a pandas DataFrame.
3.  Preprocess Data:
    - Converts binary categorical features ('mainroad', 'guestroom`, `basement`, `hotwaterheating`, `airconditioning`,`prefarea`) to numerical (1 for 'yes', 0 for 'no').
    - Applies one-hot encoding to the `furnishingstatus' categorical feature, creating new binary columns for each category (dropping the first to avoid multicollinearity).
4.  Split Data: Divides the dataset into training (80%) and testing (20%) sets using a 'random_state' for reproducibility. The features ('X') and the target variable ('price`, 'y') are separated.
5.  Train Model: Initializes and trains a linear regression model using the training data ('X_train', `y_train').
6.  Evaluate and Plot:
    - Predicts house prices on the test data (`X_test').
    - Calculates and prints the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to assess the model's performance.
    - Generates a scatter plot comparing the actual and predicted prices against the 'area' feature in the test set.
    - Prints a DataFrame showing the coefficients learned by the linear regression model for each feature, sorted in descending order.


TASK 4

1.Importing Libraries:We start by bringing in the tools we need—pandas and NumPy for handling data, matplotlib and seaborn for creating graphs, and scikit-learn for building and evaluating our machine learning model.

2.Loading the Dataset:We read the breast cancer dataset from a CSV file. Then, we clean it up a bit by removing unnecessary columns like IDs and converting the target column (diagnosis) from text labels ('M' for malignant and 'B' for benign) into numbers (1 and 0) so the model can understand it.

3.Preparing the Data:We split the dataset into two parts—one for training the model and one for testing it. Then we scale the features using standardization so that all values are on a similar range, which helps the model learn better.

4.Training the Model:We use logistic regression, a simple but powerful algorithm for binary classification, and train it on our cleaned and prepared training data.

5.Evaluating the Model:After training, we test the model’s performance on the test data. We check how well it's doing using a confusion matrix, a detailed classification report (which shows precision, recall, etc.), and the ROC-AUC score. We also visualize this performance with plots like the ROC curve.

6.Fine-Tuning and Understanding the Model:We experiment by changing the probability threshold (e.g., from 0.5 to 0.6) to see how it affects predictions. Then, we re-check performance. Lastly, we look at the sigmoid function—a key part of logistic regression—to understand how the model turns raw numbers into probabilities.

Task 5: Decision Trees and Random Forests

Objective:
Learn and implement tree-based models for classification & regression tasks.

Tools Used:
Scikit-learn, Matplotlib

Steps:
1.Load the dataset (heart.csv).
2.Train a Decision Tree Classifier and visualize it using plot_tree().
3.Analyze overfitting by comparing train/test accuracy and limiting tree depth.
4.Train a Random Forest Classifier and compare its accuracy.
5.Plot feature importances to understand influential features.
6.Perform cross-validation to evaluate model robustness.

Task 6: K-Nearest Neighbors (KNN) Classification

Objective:
Understand and apply KNN for classification tasks.

Tools Used:
Scikit-learn, Pandas, Matplotlib

Steps:

1.Load the dataset (Iris.csv).
2.Normalize features using StandardScaler.
3.Use KNeighborsClassifier to build a KNN model.
4.Perform a grid search over K (1–20) with cross-validation to find the best K.
5.Evaluate the model using accuracy and a confusion matrix.
6.Plot cross-validation accuracy vs. K.
7.Visualize decision boundaries using the first two features.











