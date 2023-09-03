# Dot-Model-Sale-Prediction

Project Description:

The "Dot Model Sale Prediction" project involves the analysis and prediction of sales based on advertising data. In this project, we have conducted various data preprocessing and exploratory data analysis (EDA) tasks to understand and prepare the dataset for modeling. Here is an overview of the key steps and insights gained from this project:

**Data Import and Understanding:**
- We imported a dataset containing information about advertising expenditures on TV, radio, newspaper, and corresponding sales figures.
- The dataset consists of 200 records and 4 columns: TV, Radio, Newspaper, and Sales.
- We verified that there were no missing values or duplicate entries in the dataset.

**Data Cleaning and Outlier Handling:**
- We identified and addressed outliers in the "Newspaper" column using the Interquartile Range (IQR) method.
- Outliers were removed to ensure the data's integrity and to prevent them from affecting model performance.

**Exploratory Data Analysis (EDA):**
- We conducted EDA to gain insights into the relationships between advertising channels (TV, Radio, Newspaper) and sales.
- Visualizations, including scatter plots and histograms, were created to visualize data distributions and correlations.

**Model Building and Evaluation:**
- We split the data into training and testing sets.
- A Linear Regression model was trained using the training data to predict sales based on advertising expenditures.
- The model's performance was evaluated using the Mean Squared Error (MSE) metric, which was found to be approximately 1.56.

**Predictive Capability:**
- We used the trained model to make predictions. For example, given advertising expenditures on TV, Radio, and Newspaper, the model can predict the expected sales.

**Model Performance Visualization:**
- We visualized the model's performance by comparing actual sales values with predicted sales values. A scatter plot with a diagonal line of fit was used to assess how well the model predictions align with actual data.

In summary, the "Dot Model Sale Prediction" project provides valuable insights into the relationship between advertising investments and sales. The trained model can be used for sales forecasting and optimizing advertising strategies to maximize revenue.
