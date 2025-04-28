#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv("C:\Downloads\Titanic-Dataset (1).csv")  # load data set
print(df.head())# print few rows


# In[9]:


print(df.info()) #print info


# In[10]:


print(df.isnull().sum()) # checking for missing value


# In[14]:


# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column if it exists
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Convert 'Sex' to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Confirm cleaning
print(df.isnull().sum())


# In[19]:


#Visualization

# Survival count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()


# In[16]:


# Quick sample to not make it heavy
sns.pairplot(df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']])
plt.show()


# In[17]:


# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()


# In[18]:


# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()


# In[ ]:




