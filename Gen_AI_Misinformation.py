# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:18:29 2025

@author: Micah Stack and Zaid Almzaian
"""

# This dataset contains 500 rows and 31 columns, with a mix of string and 
# numeric data types. The dataset has no null values, but a few unnecessary
# columns. Its variables include date and times, authors of AI post, and
# measures of the impact of potential AI misinformation. 

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats



# Data Importation and Cleanup: We imported the dataset and excluded a couple
# of columns that were unnecessary for analysis. We additionally renamed one 
# column to date. There were no null values in the dataset, so we did not have
# to fill in any information. Finally, we explored the dataset slightly using
# descrbe and info to determine if any datatypes were incorrect. 

Gen_AI = pd.read_csv("generative_ai_misinformation_dataset.csv",
                     index_col = 0, usecols=[0,1,2,3,6,7,8,9,10,12,13,
                                             15,16,17,18,19,20,21,22,23,24,25,
                                             26,27,28,29,30])

Gen_AI = Gen_AI.rename(columns = {"timestamp":"date"})

Gen_AI.head()
Gen_AI.describe()
Gen_AI.isnull().values.any()
Gen_AI.info()

Gen_AI = Gen_AI.astype({'date':'datetime64[ns]'})

Gen_AI = Gen_AI.astype({"post_id":"string", "platform":"string", "month":"string",
               "weekday":"string", "country":"string", "city":"string",
               "timezone":"string", "factcheck_verdict":"string", 
               "model_signature":"string"})

#%% Deriving New Columns

# 1. A boolean called detected AI when the detected synthetic score is above
# 0.5.

Gen_AI["detected_AI"] = Gen_AI['detected_synthetic_score'] > 0.5

# 2. A Positivty Value column where toxicity is subtracted from sentiment score
# to determine the "total positivity" of the post.

Gen_AI['total_positivity'] = Gen_AI['sentiment_score'
                                   ] - Gen_AI['toxicity_score']

# 3. This ranks authors by followers within each country.

Gen_AI['country_follower_count_rank'] = Gen_AI.groupby('country')[
    'author_followers'].rank(method='first', ascending=False)


#%% Initial Questions

# 1. What percentage of posts are detected to be AI?

Gen_AI["detected_AI"].value_counts(True)

# 48.4% of postss are detected to be AI in this study.

# 2. What is the average positivty of all posts?

Gen_AI["total_positivity"].mean()

# About -0.492 is the average positivity score, which means that these posts
# are less positve than negative.

#3. Are detected AI posts more positive than the average post?

Gen_AI.groupby("detected_AI")["total_positivity"].mean()

# Based on this calculation, AI detected posts are, on average, more positive
# than posts that were detected to be written by humans.

# 4. What platform as the most misinformation posts?

Gen_AI.groupby('platform')['is_misinformation'].value_counts("Reddit")

# Twitter has the highest percentage of posts containing misinformation.

# 5. Which platform has the highest percentage of detected AI posts and how
# many post are AI detected?

Gen_AI.groupby('platform')['detected_AI'].value_counts("Reddit")

Gen_AI_dss_upper = Gen_AI[Gen_AI['detected_synthetic_score'] > 0.5]

Gen_AI_dss_upper.groupby('platform')['detected_AI'].count()

# Telegram has the highest percentage of its posts detected as AI with 52.4% 
# and 65 total AI detected posts. 

# 6. What is the percentage of misinformative posts?

Gen_AI["is_misinformation"].value_counts(1)

# 53.6% of posts in this study were flagged for misinformation.

#%% Visualizations

## 6. Draw at least 3-4 different and meaningful visualizations using your dataset.
## Interpret and briefly explain your visualizations.

# 1. Average Detected Synthetic Score per Month

var2 = Gen_AI.groupby('month')['detected_synthetic_score'].mean()

fix,ax = plt.subplots()
ax.bar(var2.index, var2, color = 'Yellow')
ax.set_ylabel("Average DSS")
ax.set_xlabel("Month")
ax.set_xticklabels(var2.index, rotation = 90)
plt.suptitle("Average DSS per Month")

# A bar graph of the detected synthetic score based off the month of year.
# It seems that at the end of the year, the DSS rises, compared to the 
# beginning of the year.

# 2. Detected Synthetic Score per Platform

fix,ax = plt.subplots()
Gen_AI.boxplot(column = 'detected_synthetic_score', by = 'platform')
plt.ylabel("Average DSS")
plt.xlabel("Platform")
plt.title("")
plt.suptitle("Post AI Liklieness By Platform")

# A boxplot based off of the AI likliness on the posts per platform.
# It seems that Facebook has the biggest range, while Reddit has
# the highest median AI Likeliness.

# 3. Frequency of Detected Synthetic Score

fix,ax = plt.subplots()
Gen_AI['detected_synthetic_score'].hist(bins=30, color='orange', edgecolor='red')
plt.xlabel("Detected Synthetic Score")
plt.ylabel("Frequency")
plt.show
plt.suptitle("Frequency of DSS")

# A histogram based off the frequency of detected synthetic scores
# across all posts. It shows pretty inconsistent distribution, but it peaks
# around .975-1.0. 

# 4. Average Toxicity per Platform

var3 = Gen_AI.groupby('platform')['toxicity_score'].mean()

fix,ax = plt.subplots()
ax.bar(var3.index, var3, color = 'Red')
ax.set_ylabel("Average Toxicity")
ax.set_xlabel("Platform")
plt.suptitle("Average Toxicity per Platform")

# Another bar graph based off the average toxicity per platform
# By a hair, it seems facebook has the most toxic posts, while
# Reddit is the lowest



#%% Slicing and Dicing the Data

# 1. Which countries have the highest amount of misinformation posts?

Gen_AI[Gen_AI["is_misinformation"] == 1].groupby("country")['post_id'].count()

# Germany has the highest amount of misinforamtion posts at 60.

# 2. Do sources that have been flagged for misinformation
# have more or less engagement on average than other 
# posts?

fact_checked = Gen_AI.loc[:, ["factcheck_verdict", "is_misinformation", 
                                     "engagement", "detected_AI", 
                                     "readability_score"]]

# We created a subset that will be accessed later.

fact_checked.groupby(["is_misinformation"])['engagement'].mean()

# On average, misinformative posts have more engagement than truthful
# posts

# 3. Using an additional metric (factcheck_verdict) to isolate the entirely 
# misinformative posts, what percentage of these completetely incorrect 
# posts are AI generated?

false_info = fact_checked.loc[fact_checked['factcheck_verdict'].
                              str.contains("FALSE"), :]
false_info = false_info.loc[fact_checked['is_misinformation'] == 1, :]
# Uses fact_checked to create a subset catered to the question.

false_info['detected_AI'].value_counts("True")

# Surprisingly, most of the completely false posts are not detected AI 
# posts. 

# 4. Which country and city have the most detected AI posts?

Gen_AI[Gen_AI["detected_AI"] == True].groupby(["country",
                                               "city"])['post_id'].count()

# Munich, Germany has the most posts that are detected to be AI at 22.

# 5. Do verified authors contribute to lower average detected synthetic scores?

Gen_AI.groupby('author_verified')['detected_synthetic_score'].mean()

# In short, no, it is actually the opposite. Verified authors have a higher
# DSS by about 2% on average.

# 6. What is the average author's following by country?

Gen_AI.groupby('country')['author_followers'].mean()

# It seems the average author in brazil has the most followers with,
# 54.5k, while the average author in Germany has the least, with 48.5k


# 7. Which two countries containing authors with the greatest following are 
# entirely misinformative? 

topTwoPerCountry = Gen_AI.loc[Gen_AI['country_follower_count_rank'] <= 2]
lyingTopTwo = topTwoPerCountry.groupby('country')['is_misinformation'].mean()
lyingTopTwo = lyingTopTwo[lyingTopTwo == 1.0]
lyingTopTwo

# Only the United Kingdom has both of its most influencial posts cited as
# misinformation.

# 8. Out of the top two followed posts per country, what percentage are
# detected to be AI?

topTwoPerCountry['detected_AI'].value_counts(True)

# 80% of these posts (8/10) were detected to be AI.

# 9. Which cities beginning with M have produce posts with the highest
# average engagement?

highEngagementMCities = Gen_AI.loc[Gen_AI['city'].str.contains('M') &
                                   (Gen_AI['engagement'] > 5000)]
highEngagementMCities.groupby('city')['engagement'].mean()

# Manchester seems to have the highest, at 7.791k, with Mumbai having
# the lowest engagement per post, at 7.049k. The reason we filtered for 
# M was simply to incorporate the regex function into the project.

# Now, displaying the total average follower count and engagement
# broken down by Country and City for post misinfo value

# 10. What is the total breakdown of average author followers and engagement
# by country, city, and detected AI?

Gen_AI.groupby(['country', 'city',
     'detected_AI']).agg({ 'author_followers':'mean',
      'engagement':'mean'})
                          
# This gives a "bird's eye view" of the data which is divided by country, city,
# and detected AI.


#%% Advanced Analysis

# 1. Crosstabulation and Heatmap between model_signature and detected_AI.

crosstab_df = pd.crosstab(Gen_AI['model_signature'], Gen_AI['detected_AI'],
                          margins=True, margins_name='Total',
                          normalize=True)

crosstab_df

sns.heatmap(crosstab_df)

# 2. Statistics determinging accuracy.

chi2_stat, p_val, dof, ex =stats.chi2_contingency(
pd.crosstab(Gen_AI['model_signature'], Gen_AI['detected_AI']))

chi2_stat
p_val
significant= p_val <0.05
significant

# This matches my hypothesis that a) model_signature has little 
# correlation with detected_AI (which was derived from detected_synthetic
# _score) and b) it is an innaccurate measure of AI detection overall.


# 3. Crosstabulation and Heatmap between detected_AI and is_misinformation.

crosstab_df = pd.crosstab(Gen_AI['is_misinformation'], Gen_AI['detected_AI'],
                          margins=True, margins_name='Total',
                          normalize=True)

crosstab_df

sns.heatmap(crosstab_df)

# 4. Statistics determining accuracy.

chi2_stat, p_val, dof, ex =stats.chi2_contingency(
pd.crosstab(Gen_AI['is_misinformation'], Gen_AI['detected_AI']))

chi2_stat
p_val
significant= p_val <0.05
significant

# Once again, this shows that there is very little correlation between these 
# metrics, which shows that AI does not spread misinformation
# more or less than anything else; there is simply a lot of misinformation
# overall.

