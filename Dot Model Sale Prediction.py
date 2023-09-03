#!/usr/bin/env python
# coding: utf-8

# ## Import Important Library

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ## Import DataSet

# In[7]:


df = pd.read_csv(r"C:\Users\meanu\Downloads\advertising.csv - advertising.csv.csv")


# In[8]:


df.head()


# ## Data Understanding and Data Cleaning:-

# In[9]:


df.shape


# ## Find Information

# In[10]:


df.info()


# ## Finding Datatype

# In[11]:


df.dtypes


# ## Checking Duplicates Values Available Or Not:-

# In[17]:


df.duplicated().sum()


# ## Checking Null Values Available or Not :-

# In[18]:


df.isna().sum()


# ## Checking And Showing Outliers In All Column:-

# In[19]:


df.columns


# In[22]:


# Select the column containing numerical data :-
numerical_columns = df.columns

# Create box plots for each numerical column :
for column in numerical_columns:
    plt.figure(figsize=(12,8), dpi = 200)
    sns.boxplot(data= df[column])
    plt.title(f'Box plot - {column}')
    
    plt.show()


# ## Removing Outliers :-

# In[23]:


# Newspaper column has outliers and removing outliers :
q1, q2, q3 = np.percentile (df["Newspaper"], [25,50,75])
iqr = q3-q1
lower_extreme = q1=1.5*iqr
upper_extreme = q3+1.5*iqr
df= df.loc[(df["Newspaper"]>= lower_extreme) & (df["Newspaper"]<=upper_extreme)]

df


# In[24]:


df.reset_index(drop = True, inplace = True)
df


# ## Exploratory Data Analysis :-

# In[25]:


df.describe()


# In[27]:


plt.figure(figsize=(12,8), dpi = 200)
sns.scatterplot(data = df, x = df['TV'], y = df["Sales"])
plt.xlabel("TV", weight = "bold", fontsize= 12, labelpad= 10)
plt.xticks(weight = "bold")
plt.yticks(weight = "bold")

plt.show()


# In[32]:


plt.figure(figsize=(12,8), dpi = 200)
sns.scatterplot(data = df, x = df["Newspaper"], y = df["Sales"])
plt.xlabel("Newspaper", weight = "bold", fontsize = 12, labelpad = 10)
plt.ylabel("Sales", weight = "bold", fontsize = 12, labelpad = 10)
plt.xticks(weight = "bold")
plt.yticks(weight = "bold")

plt.show()


# In[34]:


plt.figure(figsize=( 12, 8), dpi = 200)
sns.scatterplot(data=df, x =df["Radio"], y = df["Sales"])
plt.xlabel("Radio", weight = "bold", fontsize = 12, labelpad= 10)
plt.ylabel("Sales", weight = "bold", fontsize = 12, labelpad= 10)
plt.xticks(weight = "bold")
plt.yticks(weight = "bold")

plt.show()


# ## Finding Correlation :-
#     

# In[35]:


plt.figure(figsize= (12,8), dpi=200)
sns.heatmap(data = df.corr(), annot = True)

plt.show()


# ## Distribution of the sales column :-

# In[36]:


plt.figure(figsize=(12,8), dpi = 200)
ax =sns.histplot(data = df, x = df["Sales"])
plt.title("Distribution Of The Sales Column", fontsize = 16 , weight = "bold" , pad= 10)
plt.xlabel("Sales", weight = "bold", fontsize = 12, labelpad = 10)
plt.ylabel("Count", weight = "bold", fontsize = 12, labelpad = 10)
plt.xticks(weight = "bold")
plt.yticks(weight = "bold")


for i in ax.containers:
    i.datavalues
    ax.bar_label(i, weight="bold")
    
    
plt.show()


# ## Model Building :-

# ### Define Dataset 

# In[38]:


x = df.drop(columns = "Sales",  axis = 1)
y = df["Sales"]


# ## Train_Test_Split :-

# In[39]:


x_train ,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=True)


# ## Training Model :-

# In[40]:


linearRegression = LinearRegression()
linearRegression.fit(x_train, y_train)


# In[42]:


y_predict = linearRegression.predict(x_test)
y_predict


# ## Mean Squared Error

# In[43]:


math.sqrt(mean_squared_error(y_predict, y_test))


# ## Prediction Value

# In[44]:


linearRegression.predict([[230.1, 37.8, 69.2]])


# ## Evaluate Model Performance :-
#     

# In[45]:


plt.figure(figsize= (12,8), dpi = 200)
sns.histplot(x=y_test)
plt.xlabel('Sales', weight ="bold", fontsize = 12, labelpad= 10)
plt.ylabel('Count', weight= "bold", fontsize = 12, labelpad = 10)
plt.xticks(weight = "bold")
plt.yticks (weight= "bold")

plt.show()


# In[52]:


plt.figure(figsize =(10,6))
sns.histplot(x=y_predict)
plt.xlabel('Sales', weight ="bold", fontsize= 12, labelpad = 10)
plt.ylabel("Count", weight = "bold", fontsize= 12, labelpad = 10)
plt.title("Distribution of Sales After Predict", fontsize = 15, weight ="bold", pad = 10)
plt.xticks(weight = "bold")
plt.yticks(weight = "bold")

plt.show()


# ## Visualize Fit Of The On The Test Set :-

# In[57]:


# The predicted values against the actual values :
plt.figure(figsize=(12,8), dpi = 200)
plt.scatter(x = y_test, y = y_predict)
plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values', weight = "bold", fontsize=12, labelpad =10)
plt.ylabel('Predicted Values', weight = "bold", fontsize=12, labelpad=10)
plt.title('Line Fit on Test set', fontsize = 15, weight ='bold', pad=10)
plt.xticks(weight ="bold")
plt.yticks(weight = "bold")

plt.show()


# In[ ]:




