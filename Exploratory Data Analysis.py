import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset from an online source
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic.head())

# Display the shape of the dataset
print("Dataset Shape:", titanic.shape)

# Get an overview of the dataset (columns, data types, missing values)
numeric_columns = titanic.select_dtypes(include=['float64', 'int64'])
print("\nDataset Info:")
print(titanic.info())

# Get basic statistics of numerical columns
print("\nSummary Statistics:")
print(titanic.describe())

# Check for missing values
print("\nMissing Values:")
print(titanic.isnull().sum())

# Drop 'Cabin' column due to too many missing values
titanic.drop(columns=['Cabin'], inplace=True)

# Fill missing 'Age' values with the median if there are any missing values
if titanic['Age'].isnull().sum() > 0:
    titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode if there are any missing values
if titanic['Embarked'].isnull().sum() > 0:
    titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Verify if missing values are handled
print("Remaining Missing Values:")
print(titanic.isnull().sum())

# Drop irrelevant columns
titanic.drop(columns=['Name', 'Ticket'], inplace=True)

# Display the updated dataset structure
print("Updated Dataset Columns:")
print(titanic.columns)

# Overall survival rate
sns.countplot(data=titanic, x='Survived')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Percentage survival
survival_rate = titanic['Survived'].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")

# Survival rate by gender
sns.countplot(data=titanic, x='Sex', hue='Survived')
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Calculate survival percentages
gender_survival = titanic.groupby('Sex')['Survived'].mean() * 100
print("Survival Rate by Gender:")
print(gender_survival)

# Survival rate by class
sns.countplot(data=titanic, x='Pclass', hue='Survived')
plt.title("Survival Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Calculate survival percentages by class
class_survival = titanic.groupby('Pclass')['Survived'].mean() * 100
print("Survival Rate by Class:")
print(class_survival)

# Distribution of age
sns.histplot(titanic['Age'], kde=True, bins=30, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Survival by age (box plot)
sns.boxplot(data=titanic, x='Survived', y='Age')
plt.title("Age vs Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# Distribution of Fare
sns.histplot(titanic['Fare'], kde=True, bins=30, color='green')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# Survival based on Fare (box plot)
sns.boxplot(data=titanic, x='Survived', y='Fare')
plt.title("Fare vs Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.show()

# Count of passengers by Embarked
sns.countplot(data=titanic, x='Embarked')
plt.title("Passenger Count by Embarked")
plt.xlabel("Port of Embarkation")
plt.ylabel("Count")
plt.show()

# Survival by Embarked
sns.countplot(data=titanic, x='Embarked', hue='Survived')
plt.title("Survival by Embarked")
plt.xlabel("Port of Embarkation")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Calculate survival rates by Embarked
embarked_survival = titanic.groupby('Embarked')['Survived'].mean() * 100

# Select only numeric columns for the correlation matrix
numeric_columns = titanic.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numeric columns
corr_matrix = numeric_columns.corr()

# Print survival rates by Embarked
print("Survival Rate by Embarked:")
print(embarked_survival)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

