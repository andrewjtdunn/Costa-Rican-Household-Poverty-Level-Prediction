 # Predicting Poverty in Costa Rican Households through Proxy Means Testing

## Team Members
[Lee-Or Bentovim](https://github.com/bentoviml), [Katherine Dumais](https://github.com/kdumais111), [Andrew Dunn](https://github.com/andrewjtdunn), [Kathryn Link-Oberstar](https://github.com/klinkoberstar)

## Project Summary
Using a Kaggle dataset from the Inter-American Development Bank, we design a machine learning model to classify household-level poverty using a Proxy Means Tests methodology. After data cleaning and collapsing the data to the household level, we use several oversampling techniques and cross validation to improve model performance given imbalances in poverty categories. After testing random forests, logistic regression, naive bayes, and k-nearest neighbors, as well as different combinations of hyperparameters, we select a logistic regression as our best performing model. We also test ensemble methods and explore using a binary poverty categorization. Finally, we note limitations of our approach and recommendations for further exploration.

## Project Report
We describe our complete approach and results in a [full report](https://github.com/andrewjtdunn/Costa-Rican-Household-Poverty-Level-Prediction/blob/main/Summary%20Report.pdf).


## Acknowledgments
Professor: Chenhao Tan

Teaching Assistant: Zander Meitus

Data Source: Inter-American Development Bank data publicly hosted on [Kaggle](https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/overview).