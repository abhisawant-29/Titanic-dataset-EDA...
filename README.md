# Titanic-dataset-EDA...
Titanic Dataset: Exploratory Data Analysis (EDA) Report

1. Introduction
The Titanic dataset is one of the most famous datasets used for binary classification tasks — predicting whether a passenger survived (1) or not (0). This project performs a complete exploratory data analysis to:

- Understand the structure and quality of the data
- Uncover relationships between variables
- Prepare for building machine learning models

2. Data Loading and Inspection
Techniques used:

.head(), .info(), .describe(), .value_counts()

Why:
.head() shows sample rows to understand what data looks like.
.info() checks datatypes and missing values.
.describe() gives basic statistical insights.
.value_counts() shows distribution of categorical variables.

Key Findings:
The dataset contains variables like Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
Missing values were found in Age, Embarked, and Cabin.
Target variable (Survived) is binary.

3. Data Cleaning
Steps Taken:

- Filled missing Age values with the median (robust to outliers).
- Filled missing Embarked values with the mode (most frequent port).
- Dropped Cabin because it had too many missing values (>75%).
- Converted Sex to numerical (male=0, female=1) for ML compatibility.
- One-hot encoded Embarked into Embarked_Q and Embarked_S.

Why:
Machine learning models cannot handle missing values or text directly.
Categorical values must be encoded numerically.

4. Data Visualization and Exploration
4.1 Pairplot
Technique: sns.pairplot()

Purpose:
Visual overview of relationships between features (scatterplots + histograms).

Observations:
Survival is visually associated with lower Pclass (1st class).
Sex strongly impacts survival (females more likely to survive).

4.2 Heatmap
Technique: sns.heatmap(df.corr())

Purpose:
Check correlations between numerical features.

Observations:
Sex has strong negative correlation with Survived (~ -0.54).
Fare positively correlates with Survived.
Pclass negatively correlates with Survived.

4.3 Histograms
Technique: sns.histplot()

Purpose:
Understand distributions of individual variables.

Observations:
Age distribution: Most passengers between 20-40 years.
Fare distribution: Skewed — most paid low fares, few paid extremely high.

4.4 Boxplots
Technique: sns.boxplot(x='Survived', y='Age/Fare', data=df)

Purpose:
Visualize spread of Age and Fare based on survival.

Observations:
Survivors tended to have paid higher fares.
Young children slightly more likely to survive, but not a very strong effect.

4.5 Scatterplots
Technique: sns.scatterplot(x='Age', y='Fare', hue='Survived')

Purpose:
Visualize the interaction between two continuous variables, colored by survival.

Observations:
Survivors cluster more around higher fares.
Some older passengers survived, but not predominantly.

5. Key Insights from EDA
Feature	Relationship to Survival	Notes
Sex	Females had higher survival	Strong predictor
Pclass	1st class had higher survival	Wealth/status mattered
Fare	Higher fare → higher survival	Possibly richer passengers were saved
Age	Very young passengers had better survival	Needs deeper modeling
Embarked	Minor differences by port	Needs careful feature engineering

6. Why EDA is Needed
EDA is critical because:

It reveals data quality issues (missing data, wrong types).
It uncovers important relationships for model building.
It helps select features (Sex, Pclass, Fare) for ML modeling.
It prevents mistakes (e.g., using messy, noisy data blindly).

Good EDA leads to higher model accuracy and better understanding.
Without proper EDA, machine learning models may learn the wrong patterns, overfit, or simply perform poorly.

📌 Conclusion
Through this detailed EDA:

We discovered critical patterns in survival rates.
We prepared the dataset for machine learning.
We are now ready to build baseline models and optimize them.
