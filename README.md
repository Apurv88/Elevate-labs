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













