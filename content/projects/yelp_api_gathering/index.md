---
title: "Eating out in Lyon - Data gathering"
date: "2022-06-01"
draft: false
excerpt: "First part of a data analysis project focused on finding a delicious pizza in my hometown of Lyon."
layout: "single"
subtitle: "Gathering restaurant ratings using the yelp fusion API."
cascade:
  featured_image: 'railway_restaurant_featured.jpg'
---

*The code and data for this project is available on [github](https://github.com/c-saade/pizza_analysis).*

## Summary

This is the first part of a data analysis project focused on finding a delicious pizza in my hometown of Lyon. Follow me as I:
- extract restaurant ratings using the yelp fusion API and the request package,
- transform the json response to a tidy dataframe,
- ensure the data integrity of variables of interest.


```python
# loading all necessary packages at once
import requests
import json
import pandas as pd
import numpy as np
```

## Gathering restaurant data from the yelp API

The yelp API allows to get 50 results at a time up to the first 1000 results of a given query.
To overcome this limitation, we can first get all the restaurant subcategories (_e.g._, italian) and then loop over all categories to get up to 1000 restaurants by categories.

### Fetching all restaurant subcategories:

I first get the categories using the 'categories' endpoint:


```python
# loading my yelp API key
key_file = open('temp_api_key.txt')
api_key = key_file.readline().rstrip('\n')
key_file.close()
# replace the above lines with your own key:
# api_key = <your api key>

# specifying the headers, endpoint and parameters:
headers = {'Authorization': 'bearer %s' % api_key}

endpoint = "https://api.yelp.com/v3/categories"

parameters = {'location': 'Lyon, FR'}

# let the request package build the url and get a response:
response = requests.get(url = endpoint, params = parameters, headers = headers)
```


```python
# transform the response into a data frame
categories_df = pd.json_normalize(response.json()['categories'])
categories_df.head()
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
      <th>title</th>
      <th>parent_aliases</th>
      <th>country_whitelist</th>
      <th>country_blacklist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3dprinting</td>
      <td>3D Printing</td>
      <td>[localservices]</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abruzzese</td>
      <td>Abruzzese</td>
      <td>[italian]</td>
      <td>[IT]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>absinthebars</td>
      <td>Absinthe Bars</td>
      <td>[bars]</td>
      <td>[CZ]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acaibowls</td>
      <td>Acai Bowls</td>
      <td>[food]</td>
      <td>[]</td>
      <td>[AR, CL, IT, MX, PL, TR]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>accessories</td>
      <td>Accessories</td>
      <td>[fashion]</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



As you can see above, I now have a dataframe with all the yelp categories. Most of them have nothing to do with food, such as the first one ('3dprinting'). To keep only categories of interest, I filter the data frame to keep only the rows whose parent categories ('parent_aliases' column) contains 'restaurants':


```python
# creating a column that states whether the row is a sub-category of 'restaurants':
categories_df['is_restaurant'] = ['restaurants' in parent for parent in categories_df['parent_aliases']]
```


```python
# filtering by the 'is_restaurant' column:
restaurants_df = categories_df[categories_df.is_restaurant]
restaurants_df.head()
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
      <th>title</th>
      <th>parent_aliases</th>
      <th>country_whitelist</th>
      <th>country_blacklist</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>afghani</td>
      <td>Afghan</td>
      <td>[restaurants]</td>
      <td>[]</td>
      <td>[MX, TR]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>african</td>
      <td>African</td>
      <td>[restaurants]</td>
      <td>[]</td>
      <td>[TR]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>andalusian</td>
      <td>Andalusian</td>
      <td>[restaurants]</td>
      <td>[ES, IT]</td>
      <td>[]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>53</th>
      <td>arabian</td>
      <td>Arabic</td>
      <td>[restaurants]</td>
      <td>[]</td>
      <td>[DK]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>59</th>
      <td>argentine</td>
      <td>Argentine</td>
      <td>[restaurants]</td>
      <td>[]</td>
      <td>[FI]</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Finding all restaurants in Lyon

We can now fetch restaurants information by looping over all categories, using the 'businesses/search' endpoint of the yelp API. I first declare some parameters by the query:


```python
key_file = open('temp_api_key.txt')
api_key = key_file.readline().rstrip('\n')
key_file.close()

headers = {'Authorization': 'bearer %s' % api_key}

endpoint = "https://api.yelp.com/v3/businesses/search"

parameters = {'location': 'Lyon, FR', # where to look
             'offset' : 0, # starting from the first result
             'limit': 50, # taking 50 results (the maximum available at a time)
             'term': 'restaurants'}
```

I then declare a list in which to store the response and loop over categories:


```python
restaurants_ratings = []

# looping over categories
for category in restaurants_df.alias:
    # specifying the category in the parameters
    parameters['categories'] = category
    # looping a second time to fetch 50 results at a time
    for offset in range(0, 1000, 50):
        parameters['offset'] = offset
        response = requests.get(url = endpoint, params = parameters, headers = headers)
        
        # we break the loop if there are no restaurants left
        if not response.json().get('businesses', False):
            break
            
        # we extend the restaurants list with the new response
        restaurants_ratings.extend(response.json()['businesses'])
```


```python
len(restaurants_ratings)
```




    4995




```python
restaurants_ratings[0]
```




    {'id': 'D3NHTerar80aeR6mlyE2mw',
     'alias': 'azur-afghan-lyon',
     'name': 'Azur Afghan',
     'image_url': 'https://s3-media1.fl.yelpcdn.com/bphoto/8i5nsqv5tbxxg8HdndPY4Q/o.jpg',
     'is_closed': False,
     'url': 'https://www.yelp.com/biz/azur-afghan-lyon?adjust_creative=qbeDf2GYKB1Prc0VgyQp0A&utm_campaign=yelp_api_v3&utm_medium=api_v3_business_search&utm_source=qbeDf2GYKB1Prc0VgyQp0A',
     'review_count': 23,
     'categories': [{'alias': 'afghani', 'title': 'Afghan'}],
     'rating': 4.0,
     'coordinates': {'latitude': 45.77502, 'longitude': 4.82875},
     'transactions': [],
     'price': '€€',
     'location': {'address1': '6 Rue Villeneuve',
      'address2': None,
      'address3': None,
      'city': 'Lyon',
      'zip_code': '69004',
      'country': 'FR',
      'state': '69',
      'display_address': ['6 Rue Villeneuve', '69004 Lyon', 'France']},
     'phone': '+33478396619',
     'display_phone': '+33 4 78 39 66 19',
     'distance': 1845.795955776875}



## Data cleaning

We gathered the data from 4995 restaurants in a json format. We can first transform it to a data frame using the 'json_normalize' function which deals fairly well with nested jsons:


```python
data = pd.json_normalize(restaurants_ratings)
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4995 entries, 0 to 4994
    Data columns (total 24 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   id                        4995 non-null   object 
     1   alias                     4995 non-null   object 
     2   name                      4995 non-null   object 
     3   image_url                 4995 non-null   object 
     4   is_closed                 4995 non-null   bool   
     5   url                       4995 non-null   object 
     6   review_count              4995 non-null   int64  
     7   categories                4995 non-null   object 
     8   rating                    4995 non-null   float64
     9   transactions              4995 non-null   object 
     10  price                     2681 non-null   object 
     11  phone                     4995 non-null   object 
     12  display_phone             4995 non-null   object 
     13  distance                  4995 non-null   float64
     14  coordinates.latitude      4992 non-null   float64
     15  coordinates.longitude     4992 non-null   float64
     16  location.address1         4987 non-null   object 
     17  location.address2         3122 non-null   object 
     18  location.address3         2730 non-null   object 
     19  location.city             4995 non-null   object 
     20  location.zip_code         4995 non-null   object 
     21  location.country          4995 non-null   object 
     22  location.state            4995 non-null   object 
     23  location.display_address  4995 non-null   object 
    dtypes: bool(1), float64(4), int64(1), object(18)
    memory usage: 902.5+ KB


We can already drop columns that don't interest us:


```python
# droping useless columns:
data.drop(['image_url',
          'is_closed', #
          'url',
          'transactions',
          'phone',
          'display_phone',
          'distance',
          'location.state' # redundant with location.zip_code
          ],
         axis = 1, inplace = True)
```

Let's check that all the restaurants are in Lyon, France:


```python
data['location.country'].value_counts()
```




    FR    4994
    IT       1
    Name: location.country, dtype: int64




```python
data['location.city'].value_counts()
```




    Lyon             3678
    Villeurbanne      429
    Bron               74
    Vénissieux         60
    Oullins            54
                     ... 
    Roanne              1
    Lyon 7Eme           1
    Saint-Paul          1
    Lyon 08             1
    Oullins Cedex       1
    Name: location.city, Length: 131, dtype: int64



It seems that one italian restaurants and quite a lot of restaurants from cities near Lyon have gotten into our data, so let's filter by city:


```python
# lowering the city field to make sure we don't exlude any restaurants due to case issues
data['location.city'] = data['location.city'].str.lower()
data = data[data['location.city'].str.find('lyon') >= 0]
```


```python
# checking that all restaurants are now in lyon
data['location.city'].value_counts()
```




    lyon                      3680
    sainte-foy-lès-lyon         11
    lyon 06                     10
    lyon 07                      9
    lyon 6eme                    9
    lyon 03                      8
    lyon 2eme                    8
    lyon 9eme                    7
    lyon 1er                     6
    sainte foy les lyon          5
    lyon 02                      5
    lyon-7e-arrondissement       4
    lyon-2e-arrondissement       4
    lyon 5eme                    4
    lyon 01                      4
    lyon 04                      3
    lyon-5e-arrondissement       3
    lyon 05                      2
    lyon 3eme                    2
    lyon-3e-arrondissement       2
    lyon  eme                    1
    lyon 08                      1
    lyon cedex 3                 1
    lyon 7eme                    1
    lyon 8eme                    1
    lyon 3 eme                   1
    lyon 09                      1
    sainte foy lès lyon          1
    Name: location.city, dtype: int64




```python
# excluding restaurants from 'sainte-foy-les-lyon' which is not in lyon
data = data[data['location.city'].str.find('sainte') == -1]
```


```python
# checking that all restaurants are now in lyon
data['location.city'].value_counts()
```




    lyon                      3680
    lyon 06                     10
    lyon 6eme                    9
    lyon 07                      9
    lyon 03                      8
    lyon 2eme                    8
    lyon 9eme                    7
    lyon 1er                     6
    lyon 02                      5
    lyon-2e-arrondissement       4
    lyon-7e-arrondissement       4
    lyon 5eme                    4
    lyon 01                      4
    lyon-5e-arrondissement       3
    lyon 04                      3
    lyon 3eme                    2
    lyon 05                      2
    lyon-3e-arrondissement       2
    lyon 08                      1
    lyon cedex 3                 1
    lyon  eme                    1
    lyon 7eme                    1
    lyon 8eme                    1
    lyon 3 eme                   1
    lyon 09                      1
    Name: location.city, dtype: int64




```python
# We can finally drop the city column, since it is redundant with zip codes
data.drop('location.city', axis = 1, inplace = True)
```

We can have a final check of our filter using zip-codes: Lyon is from 69001 to 69009, so there should be no other zipcodes:


```python
data['location.zip_code'].value_counts()
```




    69003    710
    69002    658
    69007    574
    69006    552
    69001    468
    69005    293
    69009    199
    69004    150
    69008    148
              10
    69100      3
    69000      3
    26150      2
    69300      2
    69200      1
    69363      1
    69326      1
    69800      1
    69500      1
    Name: location.zip_code, dtype: int64




```python
# dropping restaurants with zip-codes outside of Lyon
data = data[(data['location.zip_code'].isin(['69001', '69002', '69003', '69004',
                                          '69005', '69006', '69007', '69008',
                                          '69009']))]
```


```python
len(data)
```




    3752



We can then inspect missing values:


```python
data.isna().sum()
```




    id                             0
    alias                          0
    name                           0
    review_count                   0
    categories                     0
    rating                         0
    price                       1556
    coordinates.latitude           0
    coordinates.longitude          0
    location.address1              5
    location.address2           1364
    location.address3           1722
    location.zip_code              0
    location.country               0
    location.display_address       0
    dtype: int64



Missing secondary and third addresses are not an issues since most places only have a primary adress. Missing prices might be concerning if we want to later analyse ratings vs price. Lastly, we can drop the rows with a missing value in coordinates and primary address, since the only concern a few restaurants.


```python
data.dropna(subset = ['coordinates.latitude', 'coordinates.longitude', 'location.address1'], inplace = True)
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
      <th>categories</th>
      <th>rating</th>
      <th>price</th>
      <th>coordinates.latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D3NHTerar80aeR6mlyE2mw</td>
      <td>azur-afghan-lyon</td>
      <td>Azur Afghan</td>
      <td>23</td>
      <td>[{'alias': 'afghani', 'title': 'Afghan'}]</td>
      <td>4.0</td>
      <td>€€</td>
      <td>45.775020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zmk41IUwIkvO_eM0UGD7Sg</td>
      <td>sufy-lyon</td>
      <td>Sufy</td>
      <td>2</td>
      <td>[{'alias': 'indpak', 'title': 'Indian'}, {'ali...</td>
      <td>3.5</td>
      <td>NaN</td>
      <td>45.752212</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ee4wtKIBI_yTz0fJD054pg</td>
      <td>tendance-afghane-lyon</td>
      <td>Tendance Afghane</td>
      <td>1</td>
      <td>[{'alias': 'afghani', 'title': 'Afghan'}]</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>45.759540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vo0U5EcXbh7qlpdaQwZchA</td>
      <td>le-conakry-lyon</td>
      <td>Le Conakry</td>
      <td>9</td>
      <td>[{'alias': 'african', 'title': 'African'}]</td>
      <td>4.0</td>
      <td>€€</td>
      <td>45.750642</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-mFHJBuCxZJ_wJrO-o2Ypw</td>
      <td>afc-africa-food-concept-lyon</td>
      <td>AFC Africa Food Concept</td>
      <td>8</td>
      <td>[{'alias': 'african', 'title': 'African'}, {'a...</td>
      <td>3.5</td>
      <td>€€</td>
      <td>45.754336</td>
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
      <th>coordinates.longitude</th>
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
      <td>4.828750</td>
      <td>6 Rue Villeneuve</td>
      <td>None</td>
      <td>None</td>
      <td>69004</td>
      <td>FR</td>
      <td>[6 Rue Villeneuve, 69004 Lyon, France]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.864384</td>
      <td>34 rue Jeanne Hachette</td>
      <td></td>
      <td>None</td>
      <td>69003</td>
      <td>FR</td>
      <td>[34 rue Jeanne Hachette, 69003 Lyon, France]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.825560</td>
      <td>25 Rue Tramassac</td>
      <td></td>
      <td></td>
      <td>69005</td>
      <td>FR</td>
      <td>[25 Rue Tramassac, 69005 Lyon, France]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.849127</td>
      <td>112 Grande rue de la Guillotière</td>
      <td></td>
      <td></td>
      <td>69007</td>
      <td>FR</td>
      <td>[112 Grande rue de la Guillotière, 69007 Lyon,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.843469</td>
      <td>14 Grande rue de la Guillotière</td>
      <td></td>
      <td></td>
      <td>69007</td>
      <td>FR</td>
      <td>[14 Grande rue de la Guillotière, 69007 Lyon, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
##### deal with the 'categories' column !!!
```


```python
data.to_csv('data/restaurants.csv')
```
