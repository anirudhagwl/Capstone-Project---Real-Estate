#!/usr/bin/env python
# coding: utf-8

# Date - 19 May 2022
# 
# Project by - Anirudh Agarwal
# 
# Cohort - August 2021

# # Capstone Project
# 
# ## Real Estate

# ### DESCRIPTION
# 
# A banking institution requires actionable insights into mortgage-backed securities, geographic business investment, and real estate analysis. 
# 
# The mortgage bank would like to identify potential monthly mortgage expenses for each region based on monthly family income and rental of the real estate.
# 
# A statistical model needs to be created to predict the potential demand in dollars amount of loan for each of the region in the USA. Also, there is a need to create a dashboard which would refresh periodically post data retrieval from the agencies.
# 
# The dashboard must demonstrate relationships and trends for the key metrics as follows: number of loans, average rental income, monthly mortgage and owner’s cost, family income vs mortgage cost comparison across different regions. The metrics described here do not limit the dashboard to these few.

#     Variables              Description
# 
# **Second mortgage**	--> Households with a second mortgage statistics
# 
# **Home equity**	--> Households with a home equity loan statistics
# 
# **Debt** -->	Households with any type of debt statistics
# 
# **Mortgage Costs**	--> Statistics regarding mortgage payments, home equity loans, utilities, and property taxes
# 
# **Home Owner Costs** -->	Sum of utilities, and property taxes statistics
# 
# **Gross Rent** -->	Contract rent plus the estimated average monthly cost of utility features
# 
# **High school Graduation** -->	High school graduation statistics
# 
# **Population Demographics** -->	Population demographics statistics
# 
# **Age Demographics** -->	Age demographic statistics
# 
# **Household Income** -->	Total income of people residing in the household
# 
# **Family Income** -->	Total income of people related to the householder

# In[1]:


import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[2]:


training_dataframe = pd.read_csv("/Users/anirudhagarwal/Library/CloudStorage/OneDrive-Personal/Purdue DS Course/Course 8/Project 1/train.csv")
testing_dataframe = pd.read_csv("/Users/anirudhagarwal/Library/CloudStorage/OneDrive-Personal/Purdue DS Course/Course 8/Project 1/test.csv")


# In[3]:


# Importing the data set and keeping the original untouched for reference
train_df = training_dataframe.copy()
test_df = testing_dataframe.copy()


# In[4]:


train_df.head() 


# In[5]:


test_df.head()


# In[6]:


# check the number of rows and columns
train_df.shape, test_df.shape


# In[7]:


# figuring out the primary key 
# unique values/non-repeating values can be used as primary key and duplicates can be removed


print('Duplicates in training dataset :')
print(train_df.duplicated().value_counts(),'\n')

print('Duplicates in testing dataset :')
print(test_df.duplicated().value_counts(),'\n')


# In[8]:


# Removing the duplicates from the data set

train_df.drop_duplicates(keep = 'first', inplace=True)
test_df.drop_duplicates(keep = 'first', inplace=True)


# In[9]:


# check the number of rows and columns after removing duplicates
train_df.shape, test_df.shape


# In[10]:


# unique values/non-repeating values can be used as primary key 

train_df.nunique() == train_df.shape[0]


# In[11]:


test_df.nunique() == test_df.shape[0]


# #### Since the number of unique values of UID matches the number of rows,UID can be used as the primary key in the data set

# In[12]:


# checking number of unique values

train_df.nunique()


# In[13]:


# Block ID column has all missing values, and SUMLEVEL and primary each have single value.
# Hence, will remove these 3 features from both the data sets, reducing the number of features from 80 to 77

train_df.drop(columns=['BLOCKID', 'SUMLEVEL','primary'], axis = 1, inplace=True)
test_df.drop(columns=['BLOCKID', 'SUMLEVEL','primary'], axis = 1, inplace=True)

train_df.shape[1], test_df.shape[1]


# In[14]:


# Missing value treatment

train_df.isna().sum()


# In[15]:



# Fill rate of the variables

1-(train_df.isna().sum()/len(train_df))


# In[16]:


train_df['data_type'] = 'Train'
test_df['data_type'] = 'Test'


# In[17]:


# last column is data type 

train_df.head(1)


# In[18]:


# last column is data type 

test_df.head(1)


# In[19]:


combined_df = train_df.append(test_df, ignore_index=True)


# In[20]:


combined_df.head()


# In[21]:


# both data sets are combined and resulting data set has 38,838 observations

combined_df.shape


# In[22]:


# check percentage of missing values

(combined_df.isna().sum()/len(combined_df))*100


# In[23]:


col_check = combined_df.isna().sum().to_frame().reset_index()
col_check


# In[24]:


# columns with null values

null_col = col_check[col_check[0]>0]['index'].tolist()
null_col


# In[25]:


# filling the missing values with their respective feature's median, because there may be outliers in 
# the data, using mean of the data is not advisable

for i in null_col:
    combined_df[i].fillna(combined_df[i].median(), inplace=True)


# In[26]:


combined_df.isna().sum()

# No more missing values


# In columns, pop, there are some records for which the value is 0. I am removing all records with population zero as these places are not revelant for our analysis.

# In[27]:


print('Number of observations with 0 Population = ', (combined_df['pop']==0).sum())


# In[28]:


# As these observations are of no use in analysis, will remove them

combined_df = combined_df.drop(combined_df[combined_df['pop']==0].index).reset_index(drop=True)


# In[29]:


print('Number of observations with 0 Population = ', (combined_df['pop']==0).sum())


# ### Exploratory Data Analysis (EDA)

# In[30]:


# Sorting the data in decending order for second mortgage
top_second_mortgage = combined_df.sort_values(by=['second_mortgage'],ascending=False)


# In[31]:


top_second_mortgage.head()


# In[32]:


print("The top 2,500 locations where percentage of households with a second mortgage is less than 50% and percent own-ership is above 10% are: \n")

top_second_mortgage[(top_second_mortgage['second_mortgage'] <= 0.5) 
                    & (top_second_mortgage['pct_own'] > 0.1)][['state','city','place']].head(25)


# In[33]:


# Equation for bad debt
combined_df['bad_debt'] = (combined_df['second_mortgage'] + 
                          combined_df['home_equity'] - 
                          combined_df['home_equity_second_mortgage'])
        
combined_df[['bad_debt']].head(10)        


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


# Creating a pie-chart for overall debt and bad debt

overall_debt = []
debt = combined_df['debt'].sum()
overall_debt.append(debt)
bad_debt = combined_df['bad_debt'].sum()
overall_debt.append(bad_debt)


# In[36]:


overall_debt


# In[37]:


print("Pie chart for overall debt and bad debt : \n")
plt.pie(overall_debt, labels=['Debt', 'Bad_Debt'], autopct='%1.2f%%', radius=2.5)
plt.show()


# In[38]:


import seaborn as sns


# In[39]:


# Box and Whisker plots
# choosing just 10 unique cities out of total number of cities
cities = combined_df['city'].unique()[0:10]


# In[40]:


df = combined_df.loc[combined_df['city'].isin(cities)]


# In[41]:


plt.figure(figsize = (30, 15))
sns.boxplot(x = df['city'], y = df['second_mortgage'])
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 25)
plt.xlabel('City', fontsize = 25)
plt.ylabel('Second Mortgage', fontsize = 25)


# In[42]:


plt.figure(figsize = (30, 15))
sns.boxplot(x = df['city'], y = df['home_equity'])
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 25)
plt.xlabel('City', fontsize = 25)
plt.ylabel('Home Equity', fontsize = 25)


# In[43]:


plt.figure(figsize = (30, 15))
sns.boxplot(x = df['city'], y = df['debt'])
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 25)
plt.xlabel('City', fontsize = 25)
plt.ylabel('Good Debt', fontsize = 25)


# In[44]:


plt.figure(figsize = (30, 15))
sns.boxplot(x = df['city'], y = df['bad_debt'])
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 25)
plt.xlabel('City', fontsize = 25)
plt.ylabel('Bad Debt', fontsize = 25)


# In[45]:


# Collated income distribution chart

f,axs = plt.subplots(1, 3, figsize = (20, 10))
sns.histplot(combined_df['hi_mean'], color='royalblue', ax=axs[0])
sns.histplot(combined_df['family_mean'], color='cyan', ax=axs[1])
sns.histplot(combined_df['rent_mean'], color='red', ax=axs[2])


# In[46]:


# Creating new field - population density = pop/Aland
combined_df['pop_density'] = combined_df['pop'] / combined_df['ALand']
combined_df[['pop_density']].head(10)


# In[47]:


# Creating a new field - median age using the below formula
combined_df['median_age'] = (((combined_df['male_age_median']*combined_df['male_pop'])+
                              (combined_df['female_age_median']*combined_df['female_pop']))/
                             (combined_df['male_pop']+combined_df['female_pop']))
combined_df[['median_age']].head(10)


# In[48]:


# Visualizing using a bar plot

plt.figure(figsize = (25, 10))
plt.bar('state', 'median_age', data=combined_df)
plt.xlabel('State', fontsize=20)
plt.ylabel('Median Age', fontsize=20)
plt.xticks(rotation=90, fontsize=15)
plt.show()


# In[49]:


# Creating a new field - population class by dividing the population in different intervals
# 0-5000 -->class 1, 5000-10000 --> class 2, 10000-15000 -->class 3, 
# 15000-25000 -->class 4, 25000-55000 -->class 5

combined_df['pop_class'] = pd.cut(x = combined_df['pop'], 
                                  bins = [0,5000,10000,15000,25000,55000], 
                                  labels = ['1', '2', '3','4','5'])


# In[50]:


combined_df['pop_class'].value_counts()


# In[51]:


import numpy as np


# In[52]:


combined_df['pop_class']= combined_df['pop_class'].astype('int64')


# In[53]:


print('The mean value for each population class :')
print("..........................................")
for i in [1,2,3,4,5]:
        for j in ['married','separated','divorced']:
            print('Population Class:',i,'|',
                  'Mean:%.3f'%combined_df[combined_df['pop_class']==i][j].mean(),'|',
                  'Status:',j)


# In[54]:


# Visualising using bar chart
plt.figure(figsize = (25, 10))
plt.bar('pop_class', 'married', data=combined_df)
plt.xlabel('Population class', fontsize=20)
plt.ylabel('Married', fontsize=20)
plt.xticks(fontsize=15)
plt.show()


# In[55]:


plt.figure(figsize = (25, 10))
plt.bar('pop_class', 'separated', data=combined_df)
plt.xlabel('Population class', fontsize=20)
plt.ylabel('Separated', fontsize=20)
plt.xticks(fontsize=15)
plt.show()


# In[56]:


plt.figure(figsize = (25, 10))
plt.bar('pop_class', 'divorced', data=combined_df)
plt.xlabel('Population class', fontsize=20)
plt.ylabel('Divorced', fontsize=20)
plt.xticks(fontsize=15)
plt.show()


# In[57]:


print("Rent as percentage of income :")
combined_df['%_rent'] = (combined_df['rent_mean']/combined_df['hi_mean'])*100
combined_df[['%_rent']].head(10)


# In[58]:


print('Total number of states :',combined_df['state'].nunique())


# In[59]:


states = combined_df['state'].unique().tolist()
states


# In[60]:


print("'Mean rent as percentage of income per state:\n")
print("States",'\t\t % Rent\n')
for i in states:
            print(i,'=','%.3f'%combined_df[combined_df['state']==i]['%_rent'].mean(),'%')
            print("------------------------")


# In[61]:


# to find correlation and plot heatmap, I will take out features that are categorical

subset1 = combined_df.iloc[:,12:77]
subset1.head()


# In[62]:


subset1.corr()


# In[63]:


plt.figure(figsize = (75,50))
sns.heatmap(data=subset1.corr(), cmap="YlGnBu")
sns.set(font_scale=4)


# ### Data Pre-processing

# In[64]:


subset1.head()


# In[65]:


pip install factor_analyzer


# In[66]:


from factor_analyzer.factor_analyzer import FactorAnalyzer


# In[67]:


fa = FactorAnalyzer()
fa.fit(subset1, 10)


# fa = FactorAnalyzer()
# fa.fit(subset1, 10)

# In[68]:


ev, v = fa.get_eigenvalues()


# In[69]:


plt.figure(figsize = (30,15))
plt.plot(range(1, subset1.shape[1]+1), ev)
plt.xticks(np.arange(0, 70, step=5))
plt.xlabel('Number of factor')
plt.ylabel('Eigen value')
plt.show()


# In[70]:


# There is an elbow bend at 7/8, will use 8 as n


# In[71]:


n = 8
fa = FactorAnalyzer(n)
fa.fit(subset1, 10)
loads = fa.loadings_

print(loads)


# In[72]:


df1 = pd.DataFrame(loads)
df1.set_index(subset1.columns, drop=True, inplace=True)
for i in range(n):
    s = 'Factor ' + str(i+1)
    df1.rename(columns = {i : s}, inplace=True)
    
df1


# In[73]:


# Creating a dataframe of latent variables
latent_variables = combined_df[['pct_own','median_age','second_mortgage','bad_debt','hs_degree']]
latent_variables.head()


# In[74]:


latent_variables.corr()


# ## Data Modeling

# In[75]:


combined_df.head()


# In[76]:


model1 = combined_df.drop(columns=['UID','COUNTYID','STATEID','state_ab','zip_code','area_code','lat','lng','type'])


# In[77]:


# We have 3 features that are categorical, will convert them into integer using label encoder

from sklearn.preprocessing import LabelEncoder


# In[78]:


le5 = LabelEncoder()
model1['state']=le5.fit_transform(model1['state'])


# In[79]:


le6 = LabelEncoder()
model1['city']=le6.fit_transform(model1['city'])


# In[80]:


le7 = LabelEncoder()
model1['place']=le7.fit_transform(model1['place'])


# In[81]:


le8 = LabelEncoder()
model1['data_type']=le8.fit_transform(model1['data_type'])


# In[82]:


model1.head()


# In[83]:


model1_train = model1[model1['data_type']==1]
model1_test = model1[model1['data_type']==0]


# In[84]:


model1_x_train = model1_train.drop(columns=['hc_mortgage_mean']).values
model1_x_train


# In[85]:


model1_x_test = model1_test.drop(columns=['hc_mortgage_mean']).values
model1_x_test


# In[86]:


model1_x_train.shape, model1_x_test.shape


# In[87]:


model1_y_train = model1_train['hc_mortgage_mean'].values
model1_y_train


# In[88]:


model1_y_test = model1_test['hc_mortgage_mean'].values
model1_y_test


# ### Linear Regression

# In[89]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ### Creating the first model with all the features

# In[90]:


model1_lr = LinearRegression()


# In[91]:


model1_lr.fit(model1_x_train,model1_y_train)


# In[92]:


model1_y_pred = model1_lr.predict(model1_x_test)


# In[93]:


print("R² for model 1 = ",r2_score(model1_y_test, model1_y_pred))


# ### **R-squared will always increase as you add more features to the model**, even if they are unrelated to the response. Thus, selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model. Hence, will reduce the number of features

# In[94]:


import math


# In[95]:


RMSE =  math.sqrt(mean_squared_error(model1_y_test, model1_y_pred))
print('Root mean square error for model 1 = ', RMSE)


# In[96]:


sns.histplot(model1_y_pred, color="red",)


# In[97]:


combined_df.corr()


# In[98]:


""" Checking for correlation amongst variables. Variables having high correlation with another and 
variables having very low correlation with the dependent variable will be eliminated
Observations:

1.UID, County ID, State ID, zip code, area code,state ab, zip code, area code, lat, long will be removed
as these have no importance in data modeling

2. The following pairs of features have high multi-colleanrity with each other, 
will remove either one of them:
Awater-Aland, pop with male_pop/female_pop,rent_mean - rent_samples, rent_sample-rent_sample_weight, 
all rent data as either of those is closely linear to the other, rent_sample- universe_sample, 
used_samples-rent_samples, hi_mean-hi_median, hi_stdev-family_stdev, hi_sample_weight-male_age_sample_weight,
hi_samples - pop, family_mean - family_median, family_stdev-hc_mortgage_mean, 
family_sample_weight-hc_samples, family_samples-hi_mean, hc_mortgage_mean-hc_mortgage_median,
family_samples_hc_mortgage_stdev, hc_mortgage_sample_weight-hc_mortgage_samples, hc_mean-hc_median,
hc_samples-hc_stdev, home_equity_second_mortgage- second_mortgage, hc_mean-home_equity, 
second_mortgage_cdf-hi_median, home_equity_cdf-hc_mean, debt_cdf-debt, 
hs_degree- hs_degree_male/hs_degree_female, median_age - male_age_median/male_age_stdev/
male_age_sample_weight/female_age_mean/female_age_median/female_age_stdev, female_age_samples-pop,
pct_own - married/separated/ divorced, bad_debt-home_equity, pop-pop_class, %rent-rent_mean
"""


# ### Creating a second model with fewer features (Removed features which were insignificant based on correlation values)

# In[99]:


model2 = model1[['state','city','place','ALand','pop','rent_mean','rent_stdev','hi_mean',
                      'hi_sample_weight','hc_stdev','second_mortgage',
                      'debt','debt_cdf','hs_degree','median_age','pct_own','pop_density','median_age',
                       'home_equity','data_type','hc_mortgage_mean']]
model2.head()


# In[100]:


model2.corr()


# In[101]:


model2_train = model2[model2['data_type']==1]
model2_test = model2[model2['data_type']==0]


# In[102]:


model2_x_train = model2_train.drop(columns=['hc_mortgage_mean']).values
model2_x_train


# In[103]:


model2_x_test = model2_test.drop(columns=['hc_mortgage_mean']).values
model2_x_test


# In[104]:


model2_y_train = model2_train['hc_mortgage_mean'].values
model2_y_train


# In[105]:


model2_y_test = model2_test['hc_mortgage_mean'].values
model2_y_test


# In[106]:


model2_lr = LinearRegression()


# In[107]:


model2_lr.fit(model2_x_train,model2_y_train)


# In[108]:


model2_y_pred = model2_lr.predict(model2_x_test)


# In[109]:


print("R² for model 2 = ",r2_score(model2_y_test, model2_y_pred))


# In[110]:


RMSE =  math.sqrt(mean_squared_error(model2_y_test, model2_y_pred))
print('Root mean square error for model 2 = ', RMSE)


# In[111]:


sns.histplot(model2_y_pred, color="red",)


# ### R² for both model 1(98%) and model 2(80%) is high. I will not proceed to find R² for individual states (as stated in problem statement)
