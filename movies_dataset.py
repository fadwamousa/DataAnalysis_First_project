#!/usr/bin/env python
# coding: utf-8

# ## introduction

# The dataset contains 5000+ movies with some metrics that measured can be classified how successful these movies are.

# ### What Questions Are We Trying To Answer? 

# - Q1. How have the success of genres changed over time - (Revenue/Rating)?
# - Q1.1 How many movies of a particular genre have been released?
# - Q1.2 Howhave the fortunes of the genres compared over time?
# - Q2. How succesful are different genres (Revenue/Rating)?
# - Q2.1 Which genres have the largest revenue and largest budgets?
# - Q2.2 Which genres are most profitable after working out Return on Investment?
# - Q2.3 Which genres are the most popular?
# - Q3. Which Directors are the most successful (Revenue/Rating)?
# - Q4. Which Attributes indicate a movie's chances of success (Revenue/Rating)?

# ### Describe Data's General Properties

# In[32]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


# reads the data from the file - denotes as CSV, it has no header row, sets column headers
df = pd.read_csv('tmdb-movies.csv')


# In[34]:


df.head(2)


# >Now lets take a look at some of the general properties of the data:

# In[35]:


df.info()


# In[36]:


df.columns


# In[37]:


df.nunique()


# - we can see that ourdataset have 10866 entries and 21 columns

# - Some attributes look like they will be useful for our research - we know the movie, release year, director ,cast, budget, revenue, rating which are all key to our research.
#  

# - Some of them contain a few missing values such as cast and director. While there are other attributes containing many more missing values such as homepage, tagline, keywords and production_companies.

# - Columns imdb_id, homepage, tagline, overview look to be interesting but not of much value in the research required here, so may be worth dropping

# >We can now start to take a look at some of the general statistics of the data:

# In[39]:


df.describe()


# In[47]:


df.drop(['id'] , axis = 1).hist(figsize=(8,8));
plt.tight_layout();
#df.drop(['id'], axis=1).hist(figsize=( 18,18));


# In[51]:


sns.heatmap(df.drop(['id'] ,axis = 1).corr(),annot=True);


# - Looks like the distributions are showing a lot of 0 values for revenue and budget and their respective adjusted

# ## Verify Data Quality

# ### Missing Data

# In[52]:


def values_table(data):
        val = data
        val_percent = 100 * val / len(data)
        val_table = pd.concat([val, val_percent], axis=1)
        val_table_ren_columns = val_table.rename(
        columns = {0 : 'Values', 1 : '% of Total Values'})
        val_table_ren_columns = val_table_ren_columns[
            val_table_ren_columns.iloc[:,0] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        #print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"      
         #   "There are " + str(val_table_ren_columns.shape[0]) +
          #    " columns that have the values you filtered for.")
        return val_table_ren_columns


# In[54]:


values_table(df.isnull().sum())


# #### Options

# - We may want to remove null rows entirely from the dataset
# - We may want to drop the columns if they appear to be predominantly NA or not usful in our dataset
# - We may want to fill the missing values with the mean values from the dataset

# #### Decision

# 
# 
# - We will drop columns imdb_id, homepage, tagline since we identified these as low importance for our research purposes anyway.
# - We will keep keywords and production_companies as they may be interesting for research
# - We need casts, director and genres for our analysis though - we can't infer these values from the data. We'll decide to drop rows with no values for these so that it doesn't affect analysis
# 
# 

# ### Outliers

# > We noticed a lot of 0s in the histograms earlier, let's get some exact figures

# In[59]:


values_table((df == 0).sum())


# #### Options
# 

# - We may want to remove these rows entirely from the dataset

# - We may want to drop the columns if they appear to be unreliable

# - We may want to change the values with the mean values from the dataset, or another value of our choosing

# ####  Decision

# - In this case, we do not want missing values for budget, budget_adj, revenu and revenue_adj affecting the research for our analysis, so we will make these values NULL instead

# ### Duplicates 

# In[60]:


sum(df.duplicated())


# Options We may want to remove duplicate rows entirely from the dataset.

# ## Data Cleansing

# - Drop Un-Meaningful Columns

# In[61]:


df.drop(columns = ['imdb_id', 'homepage', 'tagline', 'overview'] , inplace = True)


# #### Test

# In[65]:


df.shape[1] # we had 21 , now it is 17 


# - Drop Null Columns

# In[66]:


df.dropna(subset=['revenue','revenue_adj','budget','budget_adj','runtime'] , inplace = True)


# In[76]:


values_table((df == 0).sum())


# In[77]:


#drop the null values in cast, director, genres columns
columns = ['cast', 'director', 'genres']
df.dropna(subset = columns, how='any', inplace=True)


# #### Test

# In[78]:


values_table(df.isnull().sum())


# > we have a lot of 0 in our important columns 

# In[79]:



df['revenue']     = df['revenue'].replace(0,np.nan)
df['revenue_adj'] = df['revenue_adj'].replace(0,np.nan)
df['budget']      = df['budget'].replace(0,np.nan)
df['budget_adj']  = df['budget_adj'].replace(0,np.nan)
df['runtime']     = df['runtime'].replace(0,np.nan)


# df.dropna(subset = ['revenue','revenue_adj','budget','budget_adj','runtime'] , inplace = True)

# #### Test

# In[85]:


values_table((df == 0).sum())


# - Drop duplicate records

# In[86]:


df.drop_duplicates(inplace = True)


# #### Test

# In[88]:


sum(df.duplicated())


# # Exploratory Data Analysis

# - Q1. How have the success of genres changed over time (Revenue/Rating)?

# In[91]:


df.genres.unique()


# > we are going to split the genres first 

# > we will copy the original data 

# In[92]:


df_movie = df.copy()


# In[96]:


df_movie['genres'] = df_movie['genres'].apply(lambda x : x.split("|")[0])


# In[98]:


df_movie['genres'].unique()


# - How many movies of a particular genre have been released?

# In[115]:


df_movie.groupby(['release_year','genres']).count()['id'].unstack()
df_movie.groupby('genres').count().id #the number of movies in each genres


# In[120]:


sns.set_style('darkgrid')
sns.set_palette("Set1", 20, .65)
df_movie.groupby('genres').count()['id'].sort_values(ascending = False).plot(kind = 'bar', color = "red" , figsize=(8,8));
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('The Number of Movies in each Genre');


# - Q2. How have the fortunes of the genres compared over time?

# >we are interested in vote_average & budget and revenue in particular

# In[125]:


genre_year = df_movie.groupby(['genres','release_year']).mean()


# In[126]:


df_gyBudget = genre_year.pivot_table(index=['genres'], columns=['release_year'], values='budget', aggfunc=np.mean)


# In[127]:


df_gyGross = genre_year.pivot_table(index=['genres'], columns=['release_year'], values='revenue', aggfunc=np.mean)


# In[128]:


df_gyVote = genre_year.pivot_table(index=['genres'], columns=['release_year'], values='vote_average', aggfunc=np.mean)


# In[141]:


f, [axA, axB, axC] = plt.subplots(figsize=(100, 80), nrows=3)
cmap = sns.cubehelix_palette(start=1.3, rot=1.3, as_cmap=True)
sns.heatmap(df_gyBudget, xticklabels=5, cmap=cmap, linewidths=0.05, ax=axA)
sns.heatmap(df_gyGross, xticklabels=5, cmap=cmap, linewidths=0.05, ax=axB)
sns.heatmap(df_gyVote, xticklabels=5, cmap=cmap, linewidths=0.05, ax=axC)
axA.set_title('Average Budget over Time for each Genre')
axB.set_title('Average Revenue over Time for each Genre')
axC.set_title('Average Rating Score over Time for each Genre')
axA.set_xlabel('Release Years')
axA.set_ylabel('Genres')
axB.set_xlabel('Release Years')
axB.set_ylabel('Genres')
axC.set_xlabel('Release Years')
axC.set_ylabel('Genres')
plt.show()


# - How succesful are different genres (Revenue/Rating)?

# In[144]:


df_movie.groupby('genres')[['budget','revenue','vote_average','popularity']].mean()


# In[152]:


f,ax=plt.subplots(figsize=(20, 10))
df_movie[['genres', 'budget', 'revenue']].groupby(['genres']).mean().sort_values(["revenue","budget"], ascending=False).plot(kind="bar",  ax=ax);
plt.xticks(rotation=75,fontsize=20)

ax.set(ylabel = 'Monetary Amount', title = 'Average Budget verses Average Revenue for Genres')

plt.show()


# - Which genres are the most popular?

# In[160]:


f,ax=plt.subplots(figsize=(20, 10))
df_movie[['genres', 'popularity', 'vote_average']].groupby(['genres']).mean().sort_values(["vote_average"], ascending=False).plot(kind="bar",  ax=ax);
plt.xticks(rotation=75,fontsize=20)

ax.set(ylabel = 'Rating', title = 'Average popularity and vote for Genres')

plt.show()


# - Which Directors are the most successful (Revenue/Rating)?

# In[161]:


df_director = df_movie.copy()


# In[169]:


df_director = df_director['director'].apply(lambda x:x.split("|")[0])


# In[175]:


#Create new dataframe
df_director_movies = df_movie
#split out the director field 
df_director_movies['director'] = df_director_movies['director'].apply(lambda x: x.split("|")[0])


# In[176]:


#Create our new dataframe for the Sumation of revenue for directors over time
df_director_revenue = df_director_movies.groupby(['director', 'release_year']).sum()['revenue']#.nlargest(10)
df_director_revenue = pd.DataFrame(df_director_revenue)


# In[177]:


df_director_revenue


# In[178]:


#Create new data frame for the total sumation of revenue for directors
df_director_revenue_total = df_director_revenue.groupby(['director']).sum()
df_director_revenue_total = pd.DataFrame(df_director_revenue_total)
df_director_revenue_total = df_director_revenue_total.sort_values(by = ['revenue'], ascending = False)


# In[179]:


#plot a barh graph
df_director_revenue_total[:10].plot(kind = 'bar', figsize=(13,6))

#setup the title and the labels 
plt.title("Top 10 Directors by Total Revenue",fontsize=15)
plt.xticks(rotation=75)
plt.xlabel("Director",fontsize= 18)
plt.ylabel("Total Revenue",fontsize= 20)
sns.set_style("whitegrid")


# - Which Attributes indicate a movie's chances of success (Revenue/Rating)?

# In[180]:


aux_df = df_movie[['revenue', 'budget', 'popularity', 'vote_average', 'release_year', 'runtime']]


# In[182]:


sns.heatmap(aux_df.corr(),annot=True);


# Looks like there is a positive correlation between budget and revenue, and a very slight positive correlation with release year and budget.

# In[ ]:




