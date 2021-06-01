
# coding: utf-8

# # Logistic Regression
# 
# For this project we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). 
# 
# We'll be trying to predict a classification- survival or deceased.
# 
# 
# We'll use a "semi-cleaned" version of the titanic data set.
# 
# ## Importing Libraries

# ## The Data

# In[119]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[120]:


Titanic_train = pd.read_csv('C:/Users/shmhatre/Desktop/Udemy/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')


# In[121]:


Titanic_train.head()


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ## Missing Data
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[122]:


sns.heatmap(Titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Every null values shown by yellow color. We are missing some age information and lot of onformation from cabin column.
# 
# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# Let's continue on by visualizing some more of the data! 

# In[123]:


sns.set_style('whitegrid') #setting style


# In[124]:


#To check the ratio of target varibles using countplot
sns.countplot(x='Survived',data=Titanic_train,palette='RdBu_r')


# Here, countplot shows 0 not survived v/s 1 survived. The percentage of not survived is more than the survived once.

# In[125]:


# Countplot based on sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=Titanic_train,palette='RdBu_r')


# From countplot of survived with comparision of Sex percentage, we can see the passengers who are not survived are more male and on the other side the passengers who survived, more than half are female.

# In[126]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=Titanic_train,palette='rainbow')


# From countplot of survived with class percenatage , we can see the that passenger from 3rd class are more in not survied category. 

# In[127]:


# Distribution of age of passenger on the Titanic
sns.distplot(Titanic_train['Age'].dropna(),kde=False,color='darkred',bins=30)


# #### Alternate method of plotting the same with histogram directly on the dataset and variable name (Age)

# In[128]:


#Alternate method of plotting the same with histogram directly on the dataset and varioable name
Titanic_train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# ### Exploring the other column in the dataset, Number of sibling and Spouse in the dataset (SibSp)

# In[129]:


sns.countplot(x='SibSp',data=Titanic_train)


# From Countplot on SibSp, shows the most passenger onboard neither children or a spouse on board. (probably men in the third class)

# In[130]:


# Distibution of fare column to see how much people are paying
Titanic_train['Fare'].hist(color='green',bins=40,figsize=(10,4))


# # Data 
# 
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we will find the other way to do this action by calculating the avergae age by passenger classpassenger class. For example:

# In[131]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=Titanic_train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.
# 
# We will create function to do this:
# Calculate the averge age by class using the pandas or ploting the boxplot on plotly.

# def impute_age(cols):
#     Age = cols[0]
#     Pclass = cols[1]
#     
#     if pd.isnull(Age):
# 
#         if Pclass == 1:
#             return 37
# 
#         elif Pclass == 2:
#             return 29
# 
#         else:
#             return 24
# 
#     else:
#         return Age

# We will apply this function to data set

# In[132]:


Titanic_train['Age'] = Titanic_train[['Age','Pclass']].apply(impute_age,axis=1)


# We will check the heatmap again !
# 

# In[133]:


sns.heatmap(Titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# We reaonably set the age value based on their class.
# 
# Now we will drop the Cabin column and the row in Embarked that is NaN

# In[134]:


Titanic_train.drop('Cabin',axis=1,inplace=True)


# In[135]:


Titanic_train.head()


# In[136]:


# Drop all null values
Titanic_train.dropna(inplace=True)


# In[137]:


sns.heatmap(Titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Now, we do not have any missing values

# # Converting Categorical Features
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[138]:


Titanic_train.info()


# We have two categorical variables (sex, embarked, pclass) and we need to convert to dummy varibale for ML algorithm

# In[139]:


sex = pd.get_dummies(Titanic_train['Sex'],drop_first=True)
embark = pd.get_dummies(Titanic_train['Embarked'],drop_first=True)
pclass =pd.get_dummies(Titanic_train['Pclass'],drop_first=True)


# In[140]:


Titanic_train = pd.concat([Titanic_train, sex, embark, pclass], axis =1)


# Now drop the original coulumns sex, embarked, Name and Ticket from dataset

# In[141]:


Titanic_train.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)


# We also dropped the PassengerId column.

# In[142]:


Titanic_train.drop('PassengerId', axis=1, inplace =True)


# In[143]:


Titanic_train.head()


# Now our data is ready for our model

# # Building a Logistic Regression model¶
# 
# Let's start by splitting our data into a training set and test set
# 
# # Train Test Split

# In[151]:


X = Titanic_train.drop('Survived', axis=1)
y = Titanic_train['Survived']   


# In[156]:


from sklearn.model_selection import train_test_split


# In[154]:


X_train, X_test, y_train, y_test = train_test_split(Titanic_train.drop('Survived',axis=1), 
                                                    Titanic_train['Survived'], test_size=0.30, 
                                                    random_state=101)


# # Training and Predicting

# In[159]:


from sklearn.linear_model import LogisticRegression


# In[160]:


#Creating instance of logistic regression


# In[161]:


logmodel = LogisticRegression()


# In[163]:


logmodel.fit(X_train,y_train)


# In[164]:


predictions = logmodel.predict(X_test)


# # Model Evauation
# 
# We can check precision, recall, f1-score using classification report!

# In[165]:


from sklearn.metrics import classification_report


# In[166]:


print(classification_report(y_test,predictions))
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test,predictions)


# We have the accuracy of 83 percentage, we can imrove this by using the more about feature engineering and considering other features for prediction.
