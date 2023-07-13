---
title: "Eating out in Lyon - Data analysis"
date: "2022-06-10"
draft: false
excerpt: "Second part of a data analysis project focused on finding a delicious pizza in my hometown of Lyon."
layout: "single"
subtitle: "Analyzing yelp restaurant ratings to find delicious pizza."
---

*The code and data for this project is available on [github](https://github.com/c-saade/pizza_analysis).*

## Summary

This is the second part of a data analysis project focused on finding a delicious pizza in my hometown of Lyon.
After gathering and cleaning data in the first part, I now focus on data analysis.
In the exploratory data analysis, I focus mainly on the mean price and ratings of restaurants across city districts and restaurant categories. I then go looking for outstanding restaurants — _i.e._, restaurants with a higher rating than restaurants of the same category and district — using the residuals of a linear model.

## Findings

Restaurant properties vary by city district:
- The historical center (Lyon 1 and 2) has the most restaurants, as well as the most reviews, which makes sense given its touristic appeal.
- Restaurants in fancy neighborhoods (Lyon 5 and 6) are the most expensive on average, but this is not reflected by higher ratings.
- Restaurants have better ratings on average in Lyon 7.


Here are the best pizzerias by district as identified through their residuals:

<img src="best_pizza.png" title="Best pizza places by district" width="700"/>


    
## Analysis

Let's first import some libraries and take a look at the data:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/restaurants.csv')
```


```python
data[data.columns[:8]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>alias</th>
      <th>name</th>
      <th>review_count</th>
      <th>rating</th>
      <th>price</th>
      <th>coordinates.latitude</th>
      <th>coordinates.longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D3NHTerar80aeR6mlyE2mw</td>
      <td>azur-afghan-lyon</td>
      <td>Azur Afghan</td>
      <td>23</td>
      <td>4.0</td>
      <td>€€</td>
      <td>45.775020</td>
      <td>4.828750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ee4wtKIBI_yTz0fJD054pg</td>
      <td>tendance-afghane-lyon</td>
      <td>Tendance Afghane</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>45.759540</td>
      <td>4.825560</td>
    </tr>
    <tr>
      <th>2</th>
      <td>zmk41IUwIkvO_eM0UGD7Sg</td>
      <td>sufy-lyon</td>
      <td>Sufy</td>
      <td>2</td>
      <td>3.5</td>
      <td>NaN</td>
      <td>45.752212</td>
      <td>4.864384</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vo0U5EcXbh7qlpdaQwZchA</td>
      <td>le-conakry-lyon</td>
      <td>Le Conakry</td>
      <td>9</td>
      <td>4.0</td>
      <td>€€</td>
      <td>45.750642</td>
      <td>4.849127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-mFHJBuCxZJ_wJrO-o2Ypw</td>
      <td>afc-africa-food-concept-lyon</td>
      <td>AFC Africa Food Concept</td>
      <td>8</td>
      <td>3.5</td>
      <td>€€</td>
      <td>45.754336</td>
      <td>4.843469</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data.columns[8:]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location.address1</th>
      <th>location.address2</th>
      <th>location.address3</th>
      <th>location.zip_code</th>
      <th>location.country</th>
      <th>location.display_address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6 Rue Villeneuve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69004</td>
      <td>FR</td>
      <td>['6 Rue Villeneuve', '69004 Lyon', 'France']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25 Rue Tramassac</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69005</td>
      <td>FR</td>
      <td>['25 Rue Tramassac', '69005 Lyon', 'France']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34 rue Jeanne Hachette</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69003</td>
      <td>FR</td>
      <td>['34 rue Jeanne Hachette', '69003 Lyon', 'Fran...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>112 Grande rue de la Guillotière</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69007</td>
      <td>FR</td>
      <td>['112 Grande rue de la Guillotière', '69007 Ly...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14 Grande rue de la Guillotière</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69007</td>
      <td>FR</td>
      <td>['14 Grande rue de la Guillotière', '69007 Lyo...</td>
    </tr>
  </tbody>
</table>
</div>



### Review count and ratings distribution

Let's take a look at the review count and ratings distributions:


```python
sns.displot(data = data, x = "review_count", bins = 50);
```


    
![png](output_5_0.png)
    



```python
data['price_num'] = data['price'].map({'€': 1, '€€': 2, '€€€': 3, '€€€€': 4})
```


```python
sns.countplot(data = data, x = "rating");
```


    
![png](output_7_0.png)
    


The ratings are strongly bimodal, with the bulk of the distribution centered around 4 but also numerous restaurants rated 0 stars.
The review counts are exponentially distributed, meaning that a large part of the restaurants have very few reviews.
Because ratings with few reviews might not be very informative, let's get rid of all restaurants with less than 10 reviews:


```python
data = data[data['review_count'] >= 10]
sns.countplot(data = data, x = "rating");
```


    
![png](output_9_0.png)
    


Interestingly, all ratings of 0 or 1 star only applied to restaurants with few ratings, so we now have an unimodal ratings' distribution.

### Restaurants properties by districts

Let's now take a look at how variables of interest differ between districts.
Using geopandas, I'll look at the number of restaurants, mean rating, mean price and median number of reviews by district.


```python
# importing spatial libraries
import geopandas
import geodatasets
import contextily as cx
from shapely.geometry import Point
```


```python
# importing a geojson file describing the cities district as a geo-dataframe called 'lyon_map'
# obtained from https://data.grandlyon.com/jeux-de-donnees/arrondissements-lyon/info
lyon_map = geopandas.read_file('data/adr_voie_lieu.adrarrond.geojson')

# adding zip codes to be able to merge with the data
lyon_map['location.zip_code'] = lyon_map['nomreduit'].apply(lambda x:('6900' + x[-1])).astype('int')
lyon_map
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nom</th>
      <th>nomreduit</th>
      <th>insee</th>
      <th>datemaj</th>
      <th>trigramme</th>
      <th>gid</th>
      <th>geometry</th>
      <th>location.zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lyon 1er Arrondissement</td>
      <td>Lyon 1</td>
      <td>69381</td>
      <td>1997-10-22 00:00:00+00:00</td>
      <td>LY1</td>
      <td>128</td>
      <td>POLYGON ((4.83049 45.76454, 4.83125 45.76484, ...</td>
      <td>69001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lyon 9e Arrondissement</td>
      <td>Lyon 9</td>
      <td>69389</td>
      <td>2005-07-19 00:00:00+00:00</td>
      <td>LY9</td>
      <td>181</td>
      <td>POLYGON ((4.81088 45.78099, 4.81145 45.78177, ...</td>
      <td>69009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lyon 2e Arrondissement</td>
      <td>Lyon 2</td>
      <td>69382</td>
      <td>1997-10-22 00:00:00+00:00</td>
      <td>LY2</td>
      <td>59</td>
      <td>POLYGON ((4.81782 45.72649, 4.81868 45.72660, ...</td>
      <td>69002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lyon 4e Arrondissement</td>
      <td>Lyon 4</td>
      <td>69384</td>
      <td>2003-05-12 00:00:00+00:00</td>
      <td>LY4</td>
      <td>29</td>
      <td>POLYGON ((4.81856 45.78944, 4.81817 45.78914, ...</td>
      <td>69004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lyon 3e Arrondissement</td>
      <td>Lyon 3</td>
      <td>69383</td>
      <td>2005-07-19 00:00:00+00:00</td>
      <td>LY3</td>
      <td>125</td>
      <td>POLYGON ((4.83901 45.75660, 4.83956 45.75643, ...</td>
      <td>69003</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lyon 5e Arrondissement</td>
      <td>Lyon 5</td>
      <td>69385</td>
      <td>2005-07-19 00:00:00+00:00</td>
      <td>LY5</td>
      <td>53</td>
      <td>POLYGON ((4.81353 45.74819, 4.81357 45.74823, ...</td>
      <td>69005</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lyon 7e Arrondissement</td>
      <td>Lyon 7</td>
      <td>69387</td>
      <td>2000-03-30 00:00:00+00:00</td>
      <td>LY7</td>
      <td>189</td>
      <td>POLYGON ((4.83770 45.70737, 4.83894 45.70763, ...</td>
      <td>69007</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Lyon 8e Arrondissement</td>
      <td>Lyon 8</td>
      <td>69388</td>
      <td>2011-03-03 00:00:00+00:00</td>
      <td>LY8</td>
      <td>42</td>
      <td>POLYGON ((4.84879 45.71885, 4.84881 45.71879, ...</td>
      <td>69008</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Lyon 6e Arrondissement</td>
      <td>Lyon 6</td>
      <td>69386</td>
      <td>2005-07-19 00:00:00+00:00</td>
      <td>LY6</td>
      <td>45</td>
      <td>POLYGON ((4.86994 45.76373, 4.86986 45.76384, ...</td>
      <td>69006</td>
    </tr>
  </tbody>
</table>
</div>



To obtain data by district, I group the restaurant data by 'location.zip_code' before merging it with the geodataframe 'lyon_map':


```python

# merging lyon_map with the data aggregated by zip code to get mean and median rating, price and review counts:
lyon_map = lyon_map.merge(
                data.groupby('location.zip_code')[['review_count', 'rating', 'price_num']].agg([np.mean, np.median]),
                on = 'location.zip_code',
                how = 'left')

# merging lyon_map with the data aggregated by zip code to get the restaurant count by district:
lyon_map = lyon_map.merge(
                data.groupby('location.zip_code')['id'].count().reset_index(name="restaurants_count"),
                on = 'location.zip_code',
                how = 'left')
```

I finally plot the variables of interest aggregated by district:


```python
# changing the coordinates reference system to a more visually appealing one:
lyon_map = lyon_map.to_crs(epsg = 3857)

fig, ax = plt.subplots(2, 2, figsize = (15, 12))

# plotting number of restaurants
lyon_map.plot(ax = ax[0, 0], alpha = 0.5, edgecolor = "k", column = 'restaurants_count', legend = True)
lyon_map.apply(lambda x: ax[0, 0].annotate(text=x['location.zip_code'],
                                     xy=x.geometry.centroid.coords[0],
                                     ha='center'), axis=1)
cx.add_basemap(ax[0, 0], zoom = 12, source = cx.providers.Esri.WorldGrayCanvas)
ax[0, 0].set_title('Number of restaurants')


# plotting mean rating
lyon_map.plot(ax = ax[0, 1],alpha = 0.5, edgecolor = "k",
                   column = ('rating', 'mean'), legend = True)
lyon_map.apply(lambda x: ax[0, 1].annotate(text=x['location.zip_code'],
                                     xy=x.geometry.centroid.coords[0],
                                     ha='center'), axis=1)
cx.add_basemap(ax[0, 1], zoom = 12, source = cx.providers.Esri.WorldGrayCanvas)
ax[0, 1].set_title('Mean rating')

# plotting median review count
lyon_map.plot(ax = ax[1, 0], alpha = 0.5, edgecolor = "k", column = ('review_count', 'median'), legend = True)
lyon_map.apply(lambda x: ax[1, 0].annotate(text=x['location.zip_code'],
                                     xy=x.geometry.centroid.coords[0],
                                     ha='center'), axis=1)
cx.add_basemap(ax[1, 0], zoom = 12, source = cx.providers.Esri.WorldGrayCanvas)
ax[1, 0].set_title('Median number of reviews')

# plotting mean price
lyon_map.plot(ax = ax[1, 1], alpha = 0.5, edgecolor = "k", column = ('price_num', 'mean'), legend = True)
lyon_map.apply(lambda x: ax[1, 1].annotate(text=x['location.zip_code'],
                                     xy=x.geometry.centroid.coords[0],
                                     ha='center'), axis=1)
cx.add_basemap(ax[1, 1], zoom = 12, source = cx.providers.Esri.WorldGrayCanvas)
ax[1, 1].set_title('Mean price')

# removing ticks for better visibility
for k in [0, 1]:
    for l in [0, 1]:
        ax[k, l].set_xticks([])
        ax[l, k].set_yticks([])
        
fig.tight_layout()
```


    
![png](output_16_0.png)
    


- The historical center (Lyon 1 and 2) has the most restaurants, as well as the most reviews, which makes sense given its touristic appeal.
- Restaurants in fancy neighborhoods (Lyon 5 and 6) are the most expensive on average, but this is not reflected by higher ratings.
- Restaurants have better ratings on average in Lyon 7, but this effect is rather small.

### Restaurants properties by category

Let us now take a look at how categories affect the restaurant properties.


```python
# Loading the dataframe of categories
categories_df = pd.read_csv('data/categories.csv')
```


```python
# taking a look at the distribution of the number of restaurant by category.
ax = sns.histplot(categories_df.drop('id', axis = 1).sum())
ax.set_xlabel('Number of restaurants');
```


    
![png](output_19_0.png)
    


Most categories contain very few restaurants, and are therefore not very informative. For the remainder of the analysis, I chose to keep only the most frequent categories.


```python
## dropping all categories concerning less than 100 restaurants
sorted_categories = categories_df.drop('id', axis = 1).sum().sort_values(ascending = False)
# saving a vector of the top categories
top_categories = sorted_categories[sorted_categories >= 100]
# and a vector of all other categories to filter categories_df
non_top_categories = sorted_categories[sorted_categories < 100]
```


```python
# dropping rare categories
categories_df.drop(non_top_categories.index, axis = 1, inplace = True)
```

To check how our variables of interest depend on categories, I create a dataframe of with the name of the most common categories, and then compute the restaurant count, mean rating and price and median review count for each category.


```python
# creating a dataframe of the most frequent categories ('alias') and the number of categories for each ('count')
top_categories = pd.DataFrame(top_categories)
top_categories.reset_index(inplace = True)
top_categories.columns = ['alias', 'count']
```


```python
# computing the mean rating by category
top_categories["mean_rating"] = top_categories['alias'].apply(lambda x: data['rating'][categories_df[x] > 0].mean())

# computing the median number of reviews
top_categories["median_reviews"] = top_categories['alias'].apply(lambda x: data['review_count'][categories_df[x] > 0].median())

# computing the mean price
top_categories["mean_price"] = top_categories['alias'].apply(lambda x: data['price_num'][categories_df[x] > 0].mean())

# taking a look at the resulting dataset:
top_categories
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alias</th>
      <th>count</th>
      <th>mean_rating</th>
      <th>median_reviews</th>
      <th>mean_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>french</td>
      <td>730.0</td>
      <td>3.747368</td>
      <td>30.0</td>
      <td>2.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hotdogs</td>
      <td>308.0</td>
      <td>3.366667</td>
      <td>19.0</td>
      <td>1.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pizza</td>
      <td>255.0</td>
      <td>3.561404</td>
      <td>20.0</td>
      <td>2.017544</td>
    </tr>
    <tr>
      <th>3</th>
      <td>italian</td>
      <td>174.0</td>
      <td>3.571429</td>
      <td>20.5</td>
      <td>2.115942</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sandwiches</td>
      <td>164.0</td>
      <td>3.557692</td>
      <td>20.0</td>
      <td>1.576923</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cafes</td>
      <td>159.0</td>
      <td>3.833333</td>
      <td>30.0</td>
      <td>1.974359</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bistros</td>
      <td>139.0</td>
      <td>3.957143</td>
      <td>24.0</td>
      <td>2.314286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>brasseries</td>
      <td>134.0</td>
      <td>3.346154</td>
      <td>28.0</td>
      <td>2.435897</td>
    </tr>
    <tr>
      <th>8</th>
      <td>burgers</td>
      <td>123.0</td>
      <td>3.337209</td>
      <td>34.0</td>
      <td>1.906977</td>
    </tr>
    <tr>
      <th>9</th>
      <td>japanese</td>
      <td>113.0</td>
      <td>3.661017</td>
      <td>31.0</td>
      <td>2.237288</td>
    </tr>
    <tr>
      <th>10</th>
      <td>lyonnais</td>
      <td>100.0</td>
      <td>3.701613</td>
      <td>40.0</td>
      <td>2.677419</td>
    </tr>
  </tbody>
</table>
</div>



Finally, let's take a look at the differences between categories in the form of bar charts:


```python
fig, ax = plt.subplots(2, 2, figsize = (12, 12))

sns.barplot(ax = ax[0, 0], data = top_categories, x = 'count', y = 'alias')
ax[0, 0].set_xlabel('Number of restaurants')
ax[0, 0].set_ylabel('Category')

sns.barplot(ax = ax[0, 1], data = top_categories, x = 'mean_rating', y = 'alias',
            order = top_categories.sort_values('mean_rating', ascending = False).alias)
ax[0, 1].set_xlabel('Mean rating')
ax[0, 1].set_ylabel('Category')

sns.barplot(ax = ax[1, 0], data = top_categories, x = 'median_reviews', y = 'alias',
            order = top_categories.sort_values('median_reviews', ascending = False).alias)
ax[1, 0].set_xlabel('Median number of reviews')
ax[1, 0].set_ylabel('Category')

sns.barplot(ax = ax[1, 1], data = top_categories, x = 'mean_price', y = 'alias',
            order = top_categories.sort_values('mean_price', ascending = False).alias)
ax[1, 1].set_xlabel('Mean price')
ax[1, 1].set_ylabel('Category')

fig.tight_layout();
```


    
![png](output_27_0.png)
    


## Finding the best pizza in town

I now want to identify “good” restaurants, _i.e._, restaurants that are better rated than similar restaurants. To do so, let us fit a simple linear model that predicts rating from the zip code, categories, and price of a restaurant.
Such a simple model will of course be fairly bad at predicting the rating of any individual restaurant, but its category, and price.
The residuals of the model (the difference between the true rating and predicted rating) will then give us a good understanding of how well a restaurant performs compared to other restaurants with similar properties.


```python
# building and fitting a simple linear regression model to predict the rating
from sklearn.linear_model import LinearRegression

# ratings as the target variable
y = data['rating']

# keeping zip code, price, and categories as predictor variables
X = data[['id', 'location.zip_code']].merge(categories_df, how = 'left', on = 'id')
X = pd.concat([X, pd.get_dummies(data['price']).reset_index()], axis = 1).drop(['id', 'index'], axis = 1)
X = X.values

# initializing and fitting the model
lm = LinearRegression()
lm.fit(X, y);
```


```python
# getting predictions
data['predicted_rating'] = lm.predict(X)
# computing the residuals
data['residual'] = data['rating'] - data['predicted_rating']
```


```python
# creating a data frame of the pizzerias with the highest residuals in each zip code
best_pizza = data.loc[data[categories_df["pizza"] == 1].groupby('location.zip_code')['residual'].idxmax()]

# taking a quick look at the selected pizzas:
best_pizza[['name', 'rating', 'price', "location.zip_code", 'residual']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>rating</th>
      <th>price</th>
      <th>location.zip_code</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2262</th>
      <td>Le Jardin des Pentes</td>
      <td>4.0</td>
      <td>€</td>
      <td>69001</td>
      <td>0.447680</td>
    </tr>
    <tr>
      <th>1773</th>
      <td>Casa Nobile</td>
      <td>4.5</td>
      <td>€€</td>
      <td>69002</td>
      <td>0.987304</td>
    </tr>
    <tr>
      <th>1823</th>
      <td>Le Ferrari</td>
      <td>4.5</td>
      <td>€€€</td>
      <td>69003</td>
      <td>0.957764</td>
    </tr>
    <tr>
      <th>2274</th>
      <td>Chez Puce</td>
      <td>4.5</td>
      <td>€€</td>
      <td>69004</td>
      <td>0.882967</td>
    </tr>
    <tr>
      <th>1804</th>
      <td>Al Dente</td>
      <td>4.0</td>
      <td>€€</td>
      <td>69005</td>
      <td>0.463095</td>
    </tr>
    <tr>
      <th>1786</th>
      <td>Neroliva</td>
      <td>4.5</td>
      <td>€€</td>
      <td>69006</td>
      <td>0.955026</td>
    </tr>
    <tr>
      <th>1790</th>
      <td>Le Vivaldi - Nicolo e Maria</td>
      <td>4.5</td>
      <td>€€</td>
      <td>69007</td>
      <td>0.946956</td>
    </tr>
    <tr>
      <th>2282</th>
      <td>Pizza Lina</td>
      <td>4.0</td>
      <td>€€</td>
      <td>69008</td>
      <td>0.350689</td>
    </tr>
    <tr>
      <th>1808</th>
      <td>Domeva Caffé</td>
      <td>4.5</td>
      <td>€€</td>
      <td>69009</td>
      <td>0.930817</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can plot the selected pizzerias on a map:


```python
best_pizza = geopandas.GeoDataFrame(
    best_pizza,
    geometry = geopandas.points_from_xy(best_pizza['coordinates.longitude'], best_pizza['coordinates.latitude'],
    crs = 'EPSG:4326')
)

best_pizza.to_crs(str(lyon_map.crs), inplace = True)


ax = lyon_map.plot(edgecolor = "teal", facecolor = 'none', figsize=(15, 15), linewidth=2)
best_pizza.plot(ax = ax)
best_pizza.apply(lambda x: ax.annotate(text=x['name'],
                                     xy=x.geometry.centroid.coords[0],
                                     ha='center',
                                     xytext=(-5, -15) if x['location.zip_code'] in [69005, 69007] else (-5, 20),
                                     textcoords="offset points",
                                     weight = 'bold',
                                     fontsize = 12,
                                     backgroundcolor='1'), axis=1)

cx.add_basemap(ax, zoom = 13, crs = best_pizza.crs,
               source = cx.providers.Esri.WorldTopoMap);
```


    
![png](output_33_0.png)
    

