# Bank Customer Churn Prediction

## Project Backgrounds

This project focuses on building a robust Machine Learning (ML) model to predict customer churn for a bank. Customer churn, or the rate at which customers stop doing business with an entity, is a critical metric for business success. Early and accurate prediction allows the company to implement targeted retention strategies, significantly impacting overall revenue and customer acquisition cost.

## Project Objective

To deliver a reliable classification model that can identify customers at high risk of churning. This will enable the Marketing team to execute a proactive, targeted customer retention strategy, thereby reducing the high expenses associated with customer aquisition.

## Data Source

The project used a publicly available Bank Customer Churn Dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset). The dataset contains 10,000 customer records across 10 features. These features capture both customer demographics (e.g., age and gender) and banking activities (e.g., number of products used).

## Data Inspection and Cleaning

Initial inspection for data type mismatch, missing values, duplicates, and incorrect entries revealed no significant data quality issues.

## Key Insights from Exploratory Data Analysis (EDA)

* Confirmed the target variable was imbalanced, with a churn rate of approximately 20% (80% retained, 20% exited). This necessitated the use of recall as the primary evaluation metric.

* Churn appears to be notably more common among customers between age 40-60.

* Customers who churn tend to utilize more than two banking products.

* Most customers have low account balance.

## Key Insights from Cluster Analysis

K-Means was used to identify 4 distinct customer segments that informed feature engineering. The following are observations from the clustering results:

* The customers were mainly separated by country. Specifically, all customers in Cluster 0 are from Germany, all customers in Cluster 3 are from Spain, and all customers in Clusters 1 and 2 are from France.

* The algorithm was able to seperate customers with very low account balance into a distinct cluster

* Customers in Cluster 2 have a very low account balance compared to all other customers. This shows that the model was able to capture the multimodal distribution observed in the balance column during exploratory data analysis.

## Model Development

### Splitting Method and Evaluation Metrics

The data was split into training and testing sets using a stratified sampling method to preserve the class distribution in the target variable.

Given the class imbalance (20% churn), Accuracy was deemed insufficient. The evaluation metrics used were majorly precision and recall, however recall was used for model selection and tuning. Maximizing recall was essential to minimize False Negatives (i.e., failing to identify a customer who will churn), which directly aligns with the business objective of successful retention campaigns.

### Model Training and Selection

Four models were initially trained: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

The Gradient Boosting model demonstrated the strongest initial performance, achieving a recall of 0.476. The final model was developed through the following iterative optimization steps:

1. **Hyperparameter Tuning**: The learning rate was optimized to mitigate the risk of overfitting, resulting in a slight drop in recall (0.417).

2. **Feature Engineering**: Model performance was further enhanced by incorporating K-Means distance features, which improved recall to 0.429.

3. **Optimal Threshold Tuning**: Given the class imbalance, the model's prediction threshold was tuned to maximize recall across both positive and negative classes. This final step yielded the deployed model with a performance of 0.76 Recall and 0.47 Precision on the test set.

## Business Impact Assessment

To validate the model's usefulness, its financial impact was benchmarked against a naive baseline model (predicting no churn).

Assuming the bank spends 1000 euros on customer acquisition, and 200 euros on customer retention. If the marketing team's retention campaign is 100% effective, then based on the confusion matrix on of test data, implementing the predictive model is projected to help the bank reduce the overall combined CAC and CRC by approximately 45%.

## Next Steps

The following steps are recommended to move the project toward full production and sustained impact:

* Collaborate with ML Engineers for model deployment into the production environment.

* Perform hyperparameter tunning to identify the most optimal number of clusters for the K-Means preprocessing step.

* Perform hyperparameter search to find more optimal tree and ensemble parameters for the Gradient Boosting model.

* Try other models such as K-Nearest Neighbours and Support Vector Machines.