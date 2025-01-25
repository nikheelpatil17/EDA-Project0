# Titanic Dataset EDA Project

This project involves a comprehensive Exploratory Data Analysis (EDA) of the Titanic dataset, which is a widely-used dataset for classification and survival prediction tasks. The goal of this project is to understand the dataset, clean it, analyze various features, and visualize survival trends based on different factors.

## Dataset Overview

The Titanic dataset contains information about passengers aboard the Titanic, including demographic details, ticket information, and survival status. The dataset was obtained from the following source:
- [Titanic Dataset on GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

### Key Features in the Dataset
- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Ticket fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Project Workflow

1. **Data Loading**:
   - Loaded the dataset into a Pandas DataFrame.
   - Displayed basic information like shape, head, and summary statistics.

2. **Data Cleaning**:
   - Dropped the `Cabin` column due to a high percentage of missing values.
   - Filled missing `Age` values with the median and `Embarked` values with the mode.
   - Dropped irrelevant columns like `Name` and `Ticket` to simplify analysis.

3. **Exploratory Data Analysis**:
   - Calculated survival rates based on gender, class, and embarkation port.
   - Visualized data using Seaborn and Matplotlib to identify trends and patterns:
     - Survival counts.
     - Survival by gender, class, and embarkation port.
     - Distributions of age and fare.
     - Correlation matrix for numerical features.

4. **Key Insights**:
   - Survival rates were higher for females and passengers in 1st class.
   - Passengers who paid higher fares had a better chance of survival.
   - Younger passengers had slightly higher survival rates.

## Visualizations

This project includes various visualizations:
- Count plots for survival trends by gender, class, and embarkation port.
- Histograms for age and fare distributions.
- Box plots for age and fare against survival.
- A correlation heatmap for numerical features.

## Future Improvements

- Add feature engineering (e.g., creating `FamilySize` or `IsAlone` features).
- Perform hypothesis testing to statistically validate findings.
- Extend the analysis to include predictive modeling using machine learning.

## Requirements

To run the project, install the following Python libraries:
- `pandas`
- `matplotlib`
- `seaborn`

Install the required libraries using:
```bash
pip install pandas matplotlib seaborn
