# House_price_pred_project
## House Price Prediction
### Project Overview:
This project aims to predict house prices based on various features using Multiple Linear Regression, Ridge Regression, and Lasso Regression models. The dataset includes several characteristics of the houses, such as area, number of bedrooms, bathrooms, location, and more. The performance of the models is evaluated using metrics such as R² Score, Score, and Mean Squared Error (MSE).

### Dataset:
The dataset contains information about various properties, including the following columns:

**1. Id:** Unique identifier for each house

**2. Area:** Size of the house in square feet

**3. Bedrooms:** Number of bedrooms in the house

**4. Bathrooms:** Number of bathrooms in the house

**5. Floors:** Number of floors in the house

**6. YearBuilt:** The year the house was built

**7. Location:** Location of the house (categorical)

**8. Condition:** Condition of the house (e.g., good, excellent, etc.)

**9. Garage:** Number of garage spaces available

**10. Price:** Price of the house (target variable)

**---->** Here,the independent columns 'Location','Condition' and 'Garage' are categorical datatypes which are converted into numerical data type.

**---->** The column 'Id' is not much useful so, it was removed from the data.
### Objective:
The primary goal of this project is to build predictive models to estimate house prices based on the provided features. By comparing the performance of different regression models, we aim to determine the most accurate model for predicting house prices.

### Models Used:
The following regression models were used for prediction:

**1. Multiple Linear Regression:** A basic linear regression model using all features.

**2. Ridge Regression:** A linear regression model that adds regularization (L2 penalty) to prevent overfitting.

**3. Lasso Regression:** A linear regression model with L1 regularization, which can shrink some coefficients to zero, effectively selecting features.
### Performance Metrics:
To evaluate and compare the performance of the models, the following metrics were used:

**1. R² Score:** Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

**2. Score:** Represents the model’s accuracy (higher is better).

**3. Mean Squared Error (MSE):** Measures the average squared difference between the predicted and actual values (lower is better).
### Results:
The performance of the models was compared based on the metrics mentioned above. The results indicate the effectiveness of each model in predicting house prices. Key observations from the comparison are as follows:

**1. Multiple Linear Regression:** Baseline model for comparison.

**2. Ridge Regression:** Slight improvement in accuracy by reducing overfitting.

**3. Lasso Regression:** Performed feature selection by reducing some coefficients to zero, making the model more interpretable.

### Conclusion:
This project demonstrates the effectiveness of various regression models for predicting house prices. Ridge and Lasso Regression provided improved generalization and interpretability over standard Multiple Linear Regression. Further optimization and feature engineering could enhance the predictive performance of these models.

## Contact

For any questions, feedback, or collaboration inquiries, feel free to reach out to me:

- **Email**: [jahnavikunnuru3@gmail.com](mailto:your.email@example.com)
- **LinkedIn**: [www.linkedin.com/in/jahnavi-kunnuru-370048260](https://www.linkedin.com/in/yourprofile)
