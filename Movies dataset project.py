#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Movies Dataset
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This data set contains informationabout 10,000 movies collected fromThe Movie Database (TMDb),including user ratings and revenue.
# 
# ###### Question 1  (Which genres are most popular from year to year?)
# ###### Question 2 (What properties are associated with highly rated movies?)
# ###### Question 3 (Which genres have the largest revenue and largest budgets?)
# 
# 

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[4]:


# Import necessary libraries for data analysis and visualisations 
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#read the dataset from csv file
df = pd.read_csv('tmdb-movies.csv')
df.head(2)


# In[6]:


df.info()


# ###### there are 10866 entries 21 columns

# ### number of samples 

# In[8]:


df.shape[0] 


# ### number of columns 

# In[9]:


df.shape[1] 


# ### is there any duplicated rows?

# In[10]:


sum(df.duplicated())


# ### what is the types of columns?

# In[11]:


df.dtypes


# ### is there any null values in that dataset?

# In[12]:


df.isnull().sum()


# ###### some of columns have a few missing values and some have a lot of missing values 

# In[13]:


df.nunique() #number of unique values for each column


# #### statistical 

# In[14]:


df.describe()


# 
# 
# ### Data Cleaning 

# ###### Drop the Duplicated rows

# In[15]:


df.drop_duplicates(inplace = True)


# ###### There is only one duplicated row, so i will drop this row
# 

# ### drop the columns that have more missing values in it

# In[16]:


df.drop(columns = ['homepage','tagline','keywords'] , inplace = True)


# #### These columns are not important for analysis

# In[17]:


df.columns


# In[18]:


df.drop(columns = ['production_companies'] , inplace = True)


# In[19]:


df.dropna(subset=['imdb_id'],inplace = True)


# In[20]:


df.dropna(subset=['overview'],inplace = True)


# #### drop null values for the important columns that i need it for analysis

# In[21]:


df.dropna(subset=['cast','director','genres'],inplace = True)


# In[22]:


df.columns


# ###### check out about outliers of dataset

# In[23]:


df.hist(figsize=(8,8));
plt.tight_layout()


# ###### i noticed that some important columns have a lot of zero so i will change it to NULL values instead of drop it

# In[24]:


df[df['budget'] == 0]['budget'].count()


# In[25]:


df[df['revenue'] == 0]['revenue'].count()


# In[26]:


df[df['budget_adj'] == 0]['budget_adj'].count()


# In[27]:


df[df['revenue_adj'] == 0]['revenue_adj'].count()


# In[28]:


df['revenue_adj'] = df['revenue_adj'].replace(0,np.nan)
df['revenue'] = df['revenue'].replace(0,np.nan)
df['budget_adj'] = df['budget_adj'].replace(0,np.nan)
df['budget'] = df['budget'].replace(0,np.nan)


# ###### check the columns

# In[29]:


df[df['revenue_adj'] == 0]['revenue_adj'].count()


# In[30]:


df['runtime'] = df['runtime'].replace(0,np.nan)


# In[31]:


df[df['runtime'] == 0]['runtime'].count() #count of zeros values for runtime column


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (Which genres are most popular from year to year?!)

# ###### convert the fields with two values into two rows and apply the function on that fields

# In[32]:


hb_08=df[df['genres'].str.contains('|')]


# In[33]:


df1 = hb_08.copy() # create new dataframe form copying so the original doesn't effect with any changes


# In[35]:


df1['genres'] = df1['genres'].apply(lambda x:x.split("|")[0])


# In[36]:


df1.shape


# In[37]:


df1.head(1)


# In[38]:


df1['genres'].value_counts().sort_values(ascending = False).head(5)


# ###### (Insight) so the most common Genres is Drama

# ###### let's explore the result by visualization  

# In[40]:



sns.set_style('darkgrid')

# plot data
fig, ax = plt.subplots(figsize=(18,15))

sns.set_palette("Set1", 20, .65)

# use unstack()
df1['genres'].value_counts().sort_values(ascending = False).plot(kind = 'bar' );
ax.set(xlabel='Genre', ylabel='Count', title = 'the value counts for each genre')


# ### Research Question 2  (What properties are associated with highly rated movies?)

# In[44]:


df1.plot(x = 'vote_average' , y = 'popularity',kind='scatter' , color ="red");
plt.xlabel('Vote_average')
plt.ylabel('Popularity')
plt.title('Vote Average vs Popularity');


# ###### The scatter plot above illustrates that both columns are positively correlated

# In[43]:


df1.plot(x = 'vote_average' , y = 'budget' , kind = 'scatter' , color='blue');
plt.xlabel('Vote_average')
plt.ylabel('Budget')
plt.title('Vote Average vs Budget');


# ###### (insight) The scatter plot above illustrates that both columns are positively correlated which means that movies with higher budgets tend be more highly voted by viewers

# In[46]:


df1.plot( x = 'vote_average' , y = 'revenue' , kind ='scatter' , color = 'green');
plt.xlabel('Vote_average')
plt.ylabel('Revenue')
plt.title('Vote Average vs Revenue');


# ###### (insight) The scatter plot above illustrates that both columns are positively correlated which means that movies highly voted by viewers tend be more highly revenue 

# In[48]:


df_gen = df1.groupby('genres').mean() 
#displays the mean value of all columns aggregated by genre type
df_gen.sort_values('vote_average')
df_gen['vote_average'].plot(kind = 'bar' , color = 'blue');
plt.xlabel('Genres')
plt.ylabel('Count')
plt.title('Genres vs Count');


# ### Research Question 3  (Which genres have the largest revenue and largest budgets?)

# In[54]:


df1.groupby('genres').mean()[['revenue','budget']].plot(kind = 'bar' , figsize=(12,10));
plt.xlabel('Revenue & Budget')
plt.ylabel('Count')
plt.title('The relationship between Revenue and Budget');


# ## Conclusions
# 

# In this project, I was able to analyze and identify which properties are associated with movie popularity.
# After cleaning by removing unnecessary, null, and duplicated values, I created a secondary table that broke each movie down into the separate genres it falls under.
# 
# Then I plotted a few charts to assess what will be used as the dependent variable, popularity or vote average
# Next, I plotted various other variables against vote average and found that movies with higher vote averages tend to have higher budgets and higher revenue

# In[ ]:




