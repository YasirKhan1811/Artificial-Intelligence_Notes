# Complete Data Analysis of US Accidents (2016-2023)
---
This is a countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks. The dataset currently contains approximately **7.7 million** accident records.
This Dataset can be accessed here: Sobhan Moosavi. (2023). <i>US Accidents (2016 - 2023)</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/199387

The primary goal of the project is to analyze and generate insights on the traffic accidents that took place in USA from Feb. 2016 to Mar. 2023. The first part of the analysis will examine countrywide accident events. In second part, the data will be filtered for a US city where most number of accidents have occurered, then the city data will be analyzed to produce valuable insights about accients in that city.

Throughout the analysis, the following questions will be answered:

**Countrywide Analysis:**

1. What are the top 10 U.S. states with the highest number of accidents?
2. What are the top 10 Cities with most number of accidents?
3. What is the trend of accidents by year from 2016 to 2023?
4. What are the average monthly accidents (2016-2023)?
5. Which days of the week have a higher probability of accidents?
6. What is the distribution of hourly accidents throughout the day? 
7. Are there specific hours of the day when accidents are more likely to occur?
8. What are the most frequent words in the descriptions of severity 4 accidents?
9. What are the top weather conditions that contribute to the accidents?
10. What were the most common road features during the accidents?

**Miami City Analysis:**

1. Timeseries Analysis of the accidents in Miami city, analysis by year, month, day of the week, and hour of the day.
2. Which Miami streets are the most vulnerable to accidents?
3. Which streets of Miami mostly result in 3rd and 4th level accident severities?
4. Which streets cause higher delay time?
3. Present the distribution of Accidents on the Miami City map.

**To uncover the hidden stories from our data and present them in beautiful visuals to answer the above questions, let's get Started**
<div style="page-break-after: always;"></div>

## Required Libraries
```Python
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

```python
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
%matplotlib inline

import seaborn as sns
import calendar
import plotly as pt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from pylab import *

import plotly.graph_objects as go
from nltk.corpus import stopwords

import geopandas as gpd
import geoplot
from geopy.geocoders import Nominatim

import warnings
warnings.filterwarnings('ignore')
```

## Dataset Import
---

In the first place we are going to import the dataset using Pandas module.

```python
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_March23.csv")
```

<div style="page-break-after: always;"></div>

```python
print("Size of our Dataset:", df.shape)
```

Size of our Dataset: (7728394, 46)

```python
# Set the Pandas display options to show all columns
pd.set_option('display.max_columns', None)
df.head(3)
```

## Data Cleansing
---

This dataset contains a large amount of information for analysis. However, some of the fields may be overly complex and not contribute significantly to our analysis. Before proceeding further, I plan to streamline the dataset by removing the following fields:

1. **'Id'** and **'Source'**: These fields do not provide substantial information for our analysis.
2. **'End_Lat'** and **'End_Lng'**: We already have the starting coordinates, making these fields redundant.
3. **'Airport_Code'**: Since all the data pertains to the USA, specifying the nearest airport code is unnecessary.
4. **'Country'**: As mentioned earlier, all the data is related to the USA, so this field does not add value.
5. **'Weather_Timestamp'**: We have other weather-related fields that are more relevant.
6. **'Civil_Twilight', 'Nautical_Twilight', and 'Astronomical_Twilight'**: These fields may not be directly relevant to our analysis.
7. **'Timezone'**: This information can be derived from other relevant fields.

By removing these fields, we aim to simplify the dataset, making it more focused and efficient for our analysis.

```python
# Specify the names of the columns to be dropped
columns_to_drop = ['End_Lat', 'End_Lng', 'ID', 'Source', 'Airport_Code', 'Country', 'Weather_Timestamp', 'Turning_Loop', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Timezone']

# Use the drop() method to remove the specified columns
df.drop(columns=columns_to_drop, inplace=True)
```

<div style="page-break-after: always;"></div>

**Quick overview of the Data:**

```python
from pprint import pprint
def sanity_check(df):
    pprint('-'*70)
    pprint('No. of Rows: {0[0]}        No. of Columns: {0[1]}'.format(df.shape))
    pprint('-'*70)
    
    data_profile = pd.DataFrame(df.dtypes.reset_index()).rename(columns = {'index' : 'Attribute', 0 : 'DataType'}).set_index('Attribute')
    data_profile = pd.concat([data_profile,df.isnull().sum()], axis=1).rename(columns = {0 : 'Missing Values'})
    data_profile = pd.concat([data_profile,(df.isnull().mean()*100).round(2)], axis=1).rename(columns = {0 : 'Missing %'})
    data_profile = pd.concat([data_profile,df.nunique()], axis=1).rename(columns = {0 : 'Unique Values'})
    
    pprint(data_profile)
    pprint('-'*70)

sanity_check(df) # df is our dataframe 
```

![](sanity_check01.png)

<div style="page-break-after: always;"></div>

The dataset is quite extensive, with over **7 million** entries. Consequently, removing rows associated with columns containing less than **5% missing** data won't significantly impact our analysis. Therefore, we can safely eliminate data with less than 5% missing values.

```python
df.dropna(subset=['Visibility(mi)', 'Wind_Direction', 'Description', 'Humidity(%)', 'Weather_Condition', 'Temperature(F)', 'Pressure(in)', 'Sunrise_Sunset', 'Street', 'Zipcode'], inplace=True)
```

For the columns, **Precipitation(in), Wind_Chill(F), and Wind_Speed(mph)**, the missing data is in high percentage, removing missing data from these columns would cause us to lose a lot of data (around 3 million records). Therefore, we are going to impute them with the mean values of those fields.

```python
# Calculate the mean values for each column
mean_1 = df['Precipitation(in)'].mean()
mean_2 = df['Wind_Chill(F)'].mean()
mean_3 = df['Wind_Speed(mph)'].mean()

# Impute missing values in each column with their respective means
df['Precipitation(in)'].fillna(mean_1, inplace=True)
df['Wind_Chill(F)'].fillna(mean_2, inplace=True)
df['Wind_Speed(mph)'].fillna(mean_3, inplace=True)
```
**Let's run the sanity check on the modified data.**

```python
sanity_check(df)
```
![](sanity_check02.png)

<div style="page-break-after: always;"></div>

**Remove duplicate rows**

```python
print("Number of rows:", len(df.index))
df.drop_duplicates(inplace=True)
print("Number of rows after dropping duplicates:", len(df.index))
```

> Number of rows: 7426729
>
> Number of rows after dropping duplicates: 7329850

<div style="page-break-after: always;"></div>

## Exploring Accidents: A Deep Dive into Data Insights
---

### Analysis of Accidents by States and Cities

```python
state_counts = df["State"].value_counts()
fig = go.Figure(data=go.Choropleth(locations=state_counts.index, z=state_counts.values.astype(float), locationmode="USA-states", colorscale="turbo"))

fig.update_layout(title_text="Number of Accidents by State", geo_scope="usa")
fig.show()
```

![](by_state.png)

<div style="page-break-after: always;"></div>

**What are the top 10 U.S. states with the highest number of accidents?**
```python
states = pd.DataFrame(state_counts).reset_index().sort_values('count', ascending=False)
states.rename(columns={'State':'state_code', 'count':'cases'}, inplace=True)

us_states = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

# Add a new column 'State_Name' based on 'State_Code'
states['state'] = states['state_code'].map(us_states)

# Display the updated DataFrame
states.head()
```

|  |   state_code    |   cases   |  state    |
|--|:---------------:|:---------:|:---------:|
|0|CA|1651043|California|
|1|FL|838319|Florida|
|2|TX|562295|Texas|
|3|SC|368624|South Carolina|
|4|NY|331885|New York|	

<div style="page-break-after: always;"></div>

```python
fig, ax = plt.subplots(figsize = (12,5), dpi = 80)
sns.set_style('ticks')
top_10 = states[:10]

sns.barplot(x=top_10['state'], y=top_10['cases'], palette='colorblind')
plt.title("Top 10 states with the highest number of accidents\n", fontdict = {'fontsize':16, 'color':'MidnightBlue'})
plt.ylabel("\nNumber of Accidents", fontdict = {'fontsize':12, 'color':'black'})
plt.xticks(rotation=30)
plt.xlabel(None)

total_accidents = df.shape[0]
for p in ax.patches :
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2, height + 20000,
            '{:.2f}%'.format(height/total_accidents*100),
            ha = "center", fontsize = 10, weight = 'bold', color='MidnightBlue')

# Increase the font size of the axis tick labels
sns.set(rc={'xtick.labelsize': 12, 'ytick.labelsize': 12})
# Customize Y-axis tick labels to show real numbers
def format_func(value, _):
    return f'{value:.0f}'  # Format as whole numbers
ax.yaxis.set_major_formatter(FuncFormatter(format_func))

for i in ['top', 'right']:
    ax.spines[i].set_color('white')
    ax.spines[i].set_linewidth(1.5)
plt.show()
```

![](top10_states.png)

As we can see from the map and the bar chart, California has the highest number of accidents, followed by Florida and Texas.

<div style="page-break-after: always;"></div>

**What are the top 10 Cities with most number of accidents?**
```python
cities = pd.DataFrame(df["City"].value_counts()).reset_index().sort_values(by='count',ascending=False)
cities = cities.rename(columns={'City':'city','count':'cases'})
```
```python
fig, ax = plt.subplots(figsize = (12,4), dpi = 80)
sns.set_style('ticks')

sns.barplot(x=cities[:10].city, y=cities[:10].cases, palette='colorblind')
plt.title("Top 10 Cities with most number of accidents\n", fontdict = {'fontsize':16, 'color':'MidnightBlue'})
plt.ylabel("\nNumber of Accidents", fontdict = {'fontsize':12, 'color':'black'})
plt.xlabel(None)
plt.xticks(rotation=30)

# Increase the font size of the axis tick labels
sns.set(rc={'xtick.labelsize': 12, 'ytick.labelsize': 12})

# Customize Y-axis tick labels to show real numbers
def format_func(value, _):
    return f'{value:.0f}'  # Format as whole numbers
ax.yaxis.set_major_formatter(FuncFormatter(format_func))

for i in ['top', 'right']:
    ax.spines[i].set_color('white')
    ax.spines[i].set_linewidth(1.5)

plt.show()
```

![](top10_cities.png)

<div style="page-break-after: always;"></div>

### Time Series Analysis

```python
# convert the Start_Time and End_Time attributes to datetime
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors='coerce', dayfirst=True)
df["End_Time"] = pd.to_datetime(df["End_Time"], format="mixed", errors='coerce', dayfirst=True)

# Extract year, month, weekday and day
df["Year"] = df["Start_Time"].dt.year
df["Month"] = df["Start_Time"].dt.month
df["Weekday"] = df["Start_Time"].dt.weekday
df["Day"] = df["Start_Time"].dt.day
df["Hour"] = df["Start_Time"].dt.hour
```

**How do the accidents vary by year?**
```python
year_df = pd.DataFrame(df['Year'].value_counts()).reset_index().sort_values(by='Year', ascending=True)
year = year_df.rename(columns={'Year':'year','count':'cases'})
```
```python
fig, ax = plt.subplots(figsize = (8,5), dpi = 80)
sns.set_style('ticks') # style must be one of white, dark, whitegrid, darkgrid, ticks 

# Determine the colors (as before)
colors = ['red' if val == max(year['cases']) else 'skyblue' if val == min(year['cases']) else 'lightgrey' for val in year['cases']]

sns.barplot(x=year.year, y=year.cases, palette=colors)
ax.spines[('top')].set_visible(False)
ax.spines[('right')].set_visible(False)
ax.set_xlabel(None)
ax.set_ylabel("No. of Accidents")
ax.set_title('Yearly Overview: Accidents Count and Percentage (2022-2023)\n', fontdict = {'fontsize':16 , 'color':'MidnightBlue'})
# Customize Y-axis tick labels to show real numbers
def format_func(value, _):
    return f'{value:.0f}'  # Format as whole numbers
ax.yaxis.set_major_formatter(FuncFormatter(format_func))
```

<div style="page-break-after: always;"></div>

The following code is continued from the previous page:

```python
for p in ax.patches :
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2,
            height + 20000,
            '{:.2f}%'.format(height/total_accidents*100),
            ha = "center",
            fontsize = 10, weight='bold', color='MidnightBlue')

for i in ['top','right']:
    side = ax.spines[i]
    side.set_visible(False)

plt.show()
```

![](yearly_accidents.png)

**What are the average monthly accidents (2016-2023)?**
```python
month_df = pd.DataFrame(df.Start_Time.dt.month.value_counts()).reset_index()
month = month_df.rename(columns={'Start_Time':'month#','count':'cases'}).sort_values(by='month#', ascending=True)
# adding month name as a column
month_map = {1:'Jan' , 2:'Feb' , 3:'Mar' , 4:'Apr' , 5:'May' , 6:'Jun', 7:'Jul' , 8:'Aug', 9:'Sep',10:'Oct' , 11:'Nov' , 12:'Dec'}
month['month_name'] = month['month#'].map(month_map)
```

<div style="page-break-after: always;"></div>

```python
fig, ax = plt.subplots(figsize = (12,4), dpi = 80)
sns.set_style('ticks')
# Determine the colors (as before)
colors = ['red' if val == max(month['cases']) else 'skyblue' if val == min(month['cases']) else 'lightgrey' for val in month['cases']]

sns.barplot(x=month.month_name, y=month.cases, palette=colors)
ax.set_title('Average Monthly Accidents (2022-2023)\n', fontdict = {'fontsize':16 , 'color':'MidnightBlue'})
ax.set_ylabel("\nNo. of Accidents\n", fontsize = 12)
ax.set_xlabel(None)

# Customize Y-axis tick labels to show real numbers
def format_func(value, _):
    return f'{value:.0f}'  # Format as whole numbers
ax.yaxis.set_major_formatter(FuncFormatter(format_func))

for p in ax.patches :
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2, height + 15000,
            '{:.2f}%'.format(height/total_accidents*100),
            ha = "center", fontsize = 10, weight='bold', color='MidnightBlue')

for i in ['top', 'right']:
    side = ax.spines[i]
    side.set_visible(False)
plt.show()
```

![](monthly.png)

<div style="page-break-after: always;"></div>

**Which days of the week have higher probability of accidents?**

```python
dow = pd.DataFrame(df['Start_Time'].dt.dayofweek.value_counts()).reset_index()
dow = dow.rename(columns={'Start_Time':'day_of_week', 'count':'cases'}).sort_values(by='day_of_week')
day_map = {0:'Monday' , 1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}   
dow['weekday'] = dow['day_of_week'].map(day_map)
```
```python
fig, ax = plt.subplots(figsize = (8,4), dpi = 80)
sns.set_style('ticks') 
ax=sns.barplot(y=dow.cases, x=dow.weekday, palette='pastel')
plt.title('Number of Accidents by Day of the Week\n', size=16, color='MidnightBlue')
plt.ylabel('\nAccident Cases', fontsize=12)
plt.xlabel('\nDay of the Week', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

total = df.shape[0]
for i in ax.patches:
    ax.text(i.get_x()+0.1, i.get_height()-55000,
    str(round((i.get_height()/total)*100, 2))+'%',
    va = "center", fontsize=10, weight='bold', color='MidnightBlue')

for i in ['top', 'right']:
    side = ax.spines[i]
    side.set_visible(False)
# Customize Y-axis tick labels to show real numbers
def format_func(value, _):
    return f'{value:.0f}'  # Format as whole numbers
ax.yaxis.set_major_formatter(FuncFormatter(format_func))
plt.show()
```

![](dow500dpi.png)

<div style="page-break-after: always;"></div>

In the chart above, it's evident that weekdays experience significantly more accidents compared to weekends, with weekend accident frequencies being at least 2/3 times lower. This trend may be attributed to the reduced volume of vehicles on the road during weekends.

#### What is the distribution of accidents throughout the day, and are there specific hours when accidents are more likely to occur?

```python
hour_of_day = pd.DataFrame(df['Hour'].value_counts()).reset_index().rename(columns={'Hour':'hour','count':'cases'})
hour_of_day.sort_values(by='hour', inplace=True)
```
```python
fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
sns.set_style('ticks')

colors = []
for x in hour_of_day['cases']:
    if int(hour_of_day[hour_of_day['cases'] == x]['hour']) <= 11:
        if x == max(list(hour_of_day['cases'])[:12]):
            colors.append('red')
        else:
            colors.append('skyblue')
    else:
        if x == max(list(hour_of_day['cases'])[12:]):
            colors.append('red')
        else:
            colors.append('lightgrey') 
# Create a bar plot of 'hourly_accident_rate'
sns.barplot(x=hour_of_day.hour, y=hour_of_day.cases, palette=colors)

plt.title('Hourly Accident Rate\n', size=16, color='MidnightBlue')
plt.ylabel('\nAccident Cases', fontsize=12)
plt.xlabel('\nTime of Day', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for i in ['top', 'right']:
    side = ax.spines[i]
    side.set_visible(False)
plt.show()
```

<div style="page-break-after: always;"></div>

![](hourly.png)

The early morning hours (around 7 AM) and the evening hours (around 4 PM) are associated with the highest number of accidents, with each time experienced more than **50,000** accidents on average.

### Accident Severity Analysis

#### What are the most frequent words in the descriptions of severity 4 accidents?**

We will find and list the most common words in the "description" column of accidents that have a severity level of 4 using some stopwords from the english language.

```python
stop = stopwords.words("english") + ["-"]
```
**Here is the complete breakdown of the above code:**

- stopwords is an NLTK module that provides access to stopwords for various languages, including English.
- `stopwords.words("english")` retrieves the list of English stopwords from NLTK's predefined stopwords corpus for the English language.
- `+["-"]` After obtaining the list of English stopwords, the code appends a custom word, "-" (a hyphen), to the list.

The final result is a Python list (stop) that contains both the NLTK English stopwords and the custom word "-", combined into a single list. This list can be used for text processing tasks such as text cleaning, tokenization, and removing stopwords from text data. The "-" symbol has been included in this list to treat it as a stopword, which means it can be easily removed or filtered out when processing text data.

<div style="page-break-after: always;"></div>

```python
description_s4 = df[df["Severity"] == 4]["Description"] # filter the data
# Split the description
df_words = description_s4.str.lower().str.split(expand=True).stack()
```

**Explanation of the above code:**

- `.str.lower()`: it converts all the text in the description to lowercase.
- `.str.split(expand=True)`: it splits the text in each row of the description into individual words.
- The `(expand=True)` parameter ensures that the result is returned as a DataFrame with one word per column. Each row in the resulting DataFrame will contain the words extracted from the corresponding description.
- `.stack()`: after splitting the text into words and creating a DataFrame with one word per column, this method "stacks" the DataFrame, effectively converting it back into a Series.
- The result is a Series with a multi-level index, where the first level corresponds to the original index of the descriptions in description_s4, and the second level contains the individual words from each description.


```python
# If the word is not in the stopwords list
counts = df_words[~df_words.isin(stop)].value_counts()[:10]
```

**Code explanation:**

- This code is used to count the occurrences of words in `df_words` while excluding words that are in a list of stopwords `stop`

<div style="page-break-after: always;"></div>

```python
# visualize the frequencies of the top 10 words in the description
fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
sns.set_style('ticks')
sns.barplot(x=counts.values, y=counts.index, orient="h", palette = "cividis")

ax.set_title("Top 10 words in the description of Severity 4 Accidents\n", fontsize=16, color='MidnightBlue')
ax.set_xlabel("\nFrequency of each word\n")
ax.set_ylabel(None)

for i in ['top', 'right']:
    side = ax.spines[i]
    side.set_visible(False)
    
plt.show()
```

![](severity4.png)

We can see that the most used word in the description is closed. Subsequent words are accident, due and road.

<div style="page-break-after: always;"></div>

#### Distribution of Accidents by Severity Levels

```python
s4_by_yr = df[df['Severity'] == 4][['Severity','Year']].groupby('Year').agg({'Severity': 'count'}).mean().round(0)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6))

# Calculate the percentage of each severity level
severity = df['Severity'].value_counts(normalize=True).round(2) * 100
severity.plot.pie(autopct = '%1.1f%%' , ax=ax1, colors =sns.color_palette(palette='Pastel1'), pctdistance = 0.8, explode = [.03,.03,.03,.03], textprops = {'fontsize' : 12 , 'color' : 'DarkSlateBlue'}, labels=['Severity 2','Severity 3' , 'Severity 4' , 'Severity 1'])

ax1.set_title("Percent Breakdown of Accident Severity", fontdict = {'fontsize':16 , 'color':'MidnightBlue'} )
ax1.set_ylabel(None)

s = sns.countplot(data=df[['Severity','Year']] , x = 'Year' , hue='Severity' , ax=ax2, palette = 'rainbow', edgecolor='black')

ax2.axhline(s4_by_yr[0], color='Blue', linewidth=1, linestyle='dashdot')
ax2.annotate(f"Average # of Severity 4 Accidents: {s4_by_yr[0]}", va = 'center', ha='center', color='#4a4a4a', bbox=dict(boxstyle='round', pad=0.4, facecolor='Wheat', linewidth=0), xy=(-0.5,80000))
ax2.set_title("Trend of Severity Level by Year", fontdict = {'fontsize':16 , 'color':'MidnightBlue'} )
ax2.set_ylabel("\nNo. of Accidents", fontdict = {'fontsize':16 , 'color':'MidnightBlue'} )
ax2.set_xlabel(None)

for i in ['top', 'right']:
    side = ax2.spines[i]
    side.set_visible(False)
plt.show()
```
![](severity_trend.png)

<div style="page-break-after: always;"></div>

The charts above illustrate a significant trend: approximately 80% of accidents lead to **Severity 02** injuries, it's increasing magnitude each year. Despite constituting only 20% of total accidents, **Severity 3 and 4** injuries remain a serious concern due to their proximity to fatal outcomes, emphasizing the need for continued safety measures.

### Weather Conditions Analysis

#### What are the top weather conditions that contribute to the accidents?

If we analyze the weather conditions, we can see that there are lots of them, so it's better to reduce the number of unique conditions.

```python
print("No. of Weather Conditions:", len(df["Weather_Condition"].unique()))

# To view the complete list of 142 weather descriptions, run the following code
print("\nList of unique weather conditions:", list(df["Weather_Condition"].unique()))
```

To do so, we are going to replace these **142 unique conditions** with more generic and broad contributing weather descriptions.

```python
df.loc[df["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
df.loc[df["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
df.loc[df["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
df.loc[df["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
df.loc[df["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
df.loc[df["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
df.loc[df["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
df.loc[df["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
df.loc[df["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
df.loc[df["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
df.loc[df["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan
```

<div style="page-break-after: always;"></div>

**Explanation of the above code:**

- `X["Weather_Condition"]` selects the Weather_Condition column
- `.str.contains("Thunder|T-Storm", na=False), "Weather_Condition"` uses the `.str.contains()` method to look for the Thunder or T-Storm condition in the column
- The na=False argument is used to treat missing values (NaN) as False, meaning that if a row has a missing value in the column, it won't be considered a match.
- `X.loc[...]`: is responsible for selecting the rows that meet the specified condition above. It uses boolean indexing to filter the DataFrame
- The code essentially assigns the value "Thunderstorm" to the "Weather_Condition" column in the rows, where the condition `.str.contains("Thunder|T-Storm", na=False), "Weather_Condition"` is met.

```python
wc = pd.DataFrame(df['Weather_Condition'].value_counts()).reset_index().sort_values(by='count', ascending=False)
wc.rename(columns={'Weather_Condition':'weather_condition', 'count':'frequency'}, inplace=True)
# wc stands for weather condition
```

```python
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
sns.set_style('ticks')
sns.barplot(x='frequency', y='weather_condition', data=wc, palette='cividis', orient='h')

# Add labels and title
ax.set_xlabel('\nFrequency')
ax.set_ylabel('\nWeather Condition')
ax.set_title('\nTop Weather Conditions Contributing to Accidents\n', fontsize=16, color='MidnightBlue')
plt.xticks(rotation=0)  # Adjust the rotation angle of x-axis labels

# Increase the font size of the axis tick labels
sns.set(rc={'xtick.labelsize': 10, 'ytick.labelsize': 10})

# Remove top and right spines
for i in ['top', 'right']:
    ax.spines[i].set_visible(False)
# Show the plot
plt.show()
```

<div style="page-break-after: always;"></div>

![](weather.png)

**45%** of the accidents have occured on clear days. Other top weather conditions include: Cloudy, Rain, Fog, and Snow.

### Road Features Analysis

#### What were the most common road features during the accidents?

```python
road_features = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop",
                 "Traffic_Calming", "Traffic_Signal"]

data = df[road_features].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=data.values, y=data.index, orient="h", palette='cividis')
plt.title("Most frequent road features\n", fontsize=16, color='MidnightBlue')
plt.xlabel("\nFrequency")

plt.show()
```

<div style="page-break-after: always;"></div>

![](road_features.png)

As we can see, most of the accidents occured near a traffic signal, expecially where a junction or a crossing was present. The fourth most common road feature, instead, was the presence of a nearby station, probably because of the high presence of vehicles.

___

<div style="page-break-after: always;"></div>

## Analysis of Accidents in Miami City
---

Since, Miami tops the list where most number of accidents have taken place from 2016 to 2023. Let's explore the time series data of the accidents in this city and finally we will make a map visual of the city to see which streets are the most vulnerable to accidents in Miami.

### Data Manipulation for Miami City

```python
# filter the dataframe to view data related to Miami only
miami = df[df['City'] == 'Miami']

year = miami['Year'].value_counts()
month = miami['Month'].value_counts().sort_index()

month_map = {1:'Jan' , 2:'Feb' , 3:'Mar' , 4:'Apr' , 5:'May' , 6:'Jun', 7:'Jul' , 8:'Aug', 9:'Sep',10:'Oct' , 11:'Nov' , 12:'Dec'}

hour = miami['Hour'].value_counts().sort_index()
hour_severity = miami[['Hour' , 'Severity']].groupby('Hour').agg({'Hour':'count', 'Severity' : 'mean'})

daily_accidents = miami['Weekday'].value_counts().sort_index()
day_map = {0:'Monday' , 1:'Tuesday' , 2:'Wednesday' , 3:"Thursday" , 4:'Friday' , 5:"Saturday" , 6:'Sunday'}
year_map = {x:x for x in year.index}
hour_map = {x:x for x in hour.index}

light_palette = sns.color_palette(palette='pastel')
```

### Time Series Analysis of the Accidents in Miami City

Since we are going to draw time series plots, by year, month, weekday, and hour, it's better to visualize these charts in a single window for comprehensive view.

```python
fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=(16,9))

def plot_dist(kind, text, axis, LightCoral, skyblue):
    '''
    Reusable function to plot distribution based on input time criteria
    Usage : plot_dist(kind, text, axis, red , green) - all params mandatory
    kind : 'd' for day, 'm' for month , 'y' for year, 'h' for hour
    red  : list of item to be rendered red (max)
    skyblue : list of item to be rendered skyblue (min)
    text : Text to be shownn as part of the title
    axis : Axis to plot on
    '''
```

<div style="page-break-after: always;"></div>

The following code is continued from the previous page:

```python
    if kind == 'd':
        tot, ser, map = 7, daily_accidents, day_map
    elif kind == 'm':
        tot, ser, map = 12, month ,  month_map
    elif kind == 'y':
        tot, ser, map = 8, year ,  year_map
    elif kind == 'h':
        tot, ser, map = 24, hour ,  hour_map
    day_color_map = ['AliceBlue' for _ in range(tot)]
    for l in LightCoral:
        day_color_map[l] = 'LightCoral' 
    for s in skyblue:
        day_color_map[s] = 'skyblue' 
    sns.barplot(x=ser.index.map(map) , y=ser, ax = axis, palette = day_color_map)
    axis.set_xlabel(None)
    axis.set_ylabel(None)
    axis.set_title(f'Accidents by {text}', fontdict = {'fontsize':16 , 'color':'MidnightBlue'})
    axis.grid(axis='y', linestyle='-', alpha=0.4) 
plt.subplots_adjust(wspace=0.2 , hspace = 0.4)
plt.suptitle("Timeseries Analysis of Accidents in Miami" , fontsize = 18 , color="MidnightBlue")
plot_dist('d' ,"Days of the Week", ax3,[0, 1, 2, 3, 4],[])
plot_dist('y' ,"Year", ax1,[5,6],[])
plot_dist('m' ,"Month", ax2, [0, 10, 11],[])
plot_dist('h' ,"Hour", ax4,LightCoral=[13,14,15,16,17],skyblue=[])
plt.show()
```

![](miami_timeseries1000.png)

<div style="page-break-after: always;"></div>

#### What are the most accident-prone streets in Miami?

```python
top_st = miami['Street'].value_counts().sort_values(ascending=False).head(10).index.tolist()
severity_top_st = miami[miami['Street'].isin(top_st)][['Street' , 'Severity']].groupby('Street').mean()

# add a delay_time column to the miami df
diff = miami['End_Time'] - miami['Start_Time']
miami['DelayTime'] = round(diff.dt.seconds/3600,1)
top_st_delay = miami[miami['Street'].isin(top_st)][['Street' , 'DelayTime']] .groupby('Street').mean()
```

```python
fig, (ax,ax1,ax2) = plt.subplots(3,1,figsize=(16,10), sharex=True)

sns.countplot(data = miami[miami['Street'].isin(top_st)][['Street','Severity']], x='Street',ax=ax2, palette='Pastel2')
plt.xticks(rotation=30)
ax1.plot(severity_top_st, color='CornFlowerBlue', label='Severity',linewidth=3, linestyle='solid', marker='.',markersize=18, markerfacecolor='w',markeredgecolor='b',markeredgewidth='2')
ax.plot(top_st_delay, color='LightCoral', label='Severity',linewidth=3, linestyle='solid',marker='*', markersize=18, markerfacecolor='w',markeredgecolor='b',markeredgewidth='2')
ax.spines[('top')].set_visible(False)
ax.spines[('right')].set_visible(False)
ax1.spines[('right')].set_visible(False)
ax2.spines[('right')].set_visible(False)
ax2.set_xlabel("Miami Streets", fontdict = {'fontsize':14 , 'color':'Teal'} )
ax2.set_ylabel("No. of Accidents", fontdict = {'fontsize':12 , 'color':'MidnightBlue'})
ax1.set_ylabel("Severity of Accidents", fontdict = {'fontsize':12 , 'color':'MidnightBlue'})
ax.set_ylabel("Avg. Delay Times (Hours)", fontdict = {'fontsize':12 , 'color':'MidnightBlue'})
ax.set_title('Accidents on Top Miami Streets, Severity and Delay', fontdict = {'fontsize':16 , 'color':'MidnightBlue'})
ax1.legend(loc=(0.01,0.8))
ax.legend(loc=(0.01,0.8))
ax.grid(axis='x', linestyle='-', alpha=0.4) 
ax1.grid(axis='x', linestyle='-', alpha=0.4) 
ax2.grid(axis='x', linestyle='-', alpha=0.4) 

plt.show()
```
<div style="page-break-after: always;"></div>

![](streets.PNG)

<div style="page-break-after: always;"></div>

## Street View Analysis on Miami Map
---

```python
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)
fig = px.density_mapbox(miami, lat='Start_Lat', lon='Start_Lng', z='Severity', hover_name='Street', radius=5, center=dict(lat=miami['Start_Lat'].median(), lon=miami['Start_Lng'].median()), zoom=12 mapbox_style="open-street-map", height=900)
fig.show()
```

![](street_view.png)

***
---
___