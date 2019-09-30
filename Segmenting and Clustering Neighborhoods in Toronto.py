#!/usr/bin/env python
# coding: utf-8

# # Import All the Libraries

# In[1]:


import numpy as np
import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# # Install Additional Packages for web scrapping 

# In[2]:


pip install BeautifulSoup4 requests


# In[3]:


pip install lxml


# Import additional libraries

# In[4]:


import requests
from bs4 import BeautifulSoup
import lxml
import re


# # Read the webpage and extract the RAW table information

# In[5]:


url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')
source = BeautifulSoup(url.text, 'html.parser')
table = source.find_all('table',class_="wikitable sortable")


# Parse through the table content and extract table row i.e. tags with "td"

# In[6]:


for row in table:
    row_td = row.find_all('td')
    str_cells = str(row_td)
    clean = re.compile('<.*?>')
    clean2 = (re.sub(clean,'',str_cells))


# Cleanse the Extract row information and prepare the list

# In[7]:


clean2 = clean2.replace('[','')
clean2 = clean2.replace(']','')
clean2 = clean2.replace('\n, ','\n')
clean2 = clean2.replace('"',"'")
lines = clean2.split('\n')
lines = lines[:-1]


# # Load the table information into DataFrame

# Initialize the Dataframe

# In[8]:


column_names = ['PostalCode','Borough','Neighborhood'] 
# instantiate the dataframe
df = pd.DataFrame(columns=column_names)


# Load the row information into Data Frame

# In[9]:


for data in lines:
    line = data.split(',')
    PostalCode = line[0] 
    Borough = line[1]
    Neighborhood = line[2]    
   
    df = df.append({'PostalCode': PostalCode,
                    'Borough': Borough,
                    'Neighborhood': Neighborhood}, ignore_index=True)


# Check the Size of the DataFrame before any processing

# In[10]:


df.shape[0]


# Discarding the rows where Borough doesn't have any value i.e. Not Assigned

# In[11]:


df = df[df.Borough !=' Not assigned']


# More than one neighborhood can exist in one postal code area. For example, in the table on the Wikipedia page, you will notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. Combining these rows into one row with the neighborhoods separated with a comma

# In[12]:


#df=df.groupby(['PostalCode','Borough'])['Neighborhood'].apply(','.join)
#df.groupby(['PostalCode','Borough']).agg(concat_ws(',', collect_list("Neighborhood")))
df1 = df.groupby(['PostalCode','Borough'])['Neighborhood'].apply(lambda tags: ','.join(tags))
df1 = df1.to_frame()
df1 = df1.reset_index()


# Selecting rows where Neighborhood is not assigned

# In[13]:


df1[df1.Neighborhood == ' Not assigned']


# Replacing such Neighborhood value with correspodning Borough

# In[14]:


df1['Neighborhood'][df1.Neighborhood == ' Not assigned']=df1['Borough'][df1.Neighborhood == ' Not assigned']


# verifying the data

# In[15]:


df1[df1.Neighborhood == ' Not assigned']


# In[16]:


df1[df1.PostalCode == "M7A"]


# The size of the final Data Frame

# In[17]:


df1.shape


# # Appending Latitude,Longitude information for each postal code

# Reading the csv file which contains postal code wise Latitude, Longitude information

# In[18]:


df3=pd.read_csv('http://cocl.us/Geospatial_data')


# Renaming the Postal code column name to match with our data frame

# In[19]:


df3.rename(columns={'Postal Code': 'PostalCode'}, inplace=True)
df3.head()


# Concatenate 2 data frames to have consolidated data frame 

# In[20]:


toronto_data = pd.concat([df1, df3], sort=True, ignore_index=True, axis=1)


# Lets check the data

# In[21]:


toronto_data.head()


# Dropping unwanted columns and renaming the existing one

# In[22]:


toronto_data.drop(3, axis=1 , inplace=True)
toronto_data.rename(columns={0: 'PostalCode',1: 'Borough',2: 'Neighborhood',4: 'Latitude',5: 'Longitude'}, inplace=True)
toronto_data.head()


# # Use geopy library to get the latitude and longitude values of Toronto.

# In[23]:


address = 'Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto City are {}, {}.'.format(latitude, longitude))


# # Create a map of Toronto with neighborhoods superimposed on top

# In[24]:


# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Borough'], toronto_data['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# # Define Foursquare Credentials and Version

# In[25]:


CLIENT_ID = 'LOZ5UGC4Q1ERRNBQXNWXGEFU3JXBZXSIFEDMNVUL1BQD32TQ' # your Foursquare ID
CLIENT_SECRET = 'UMLEYDFRMEMXOTHQTZDIMDRX5FTHHTKIMWGJI3C3QBPNNNJD' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# # Explore Neighborhoods in Manhattan

# # Creat a function

# In[26]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# # Create a new data frame using above function

# In[28]:


# type your answer here

toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[29]:


print(toronto_venues.shape)
toronto_venues.head()


# Let's check how many venues were returned for each neighborhood

# In[30]:


toronto_venues.groupby('Neighborhood').count()


# # Analyze Each Neighborhood

# In[31]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# And let's examine the new dataframe size.

# In[32]:


toronto_onehot.shape


# Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[33]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# lets confirm the new size

# In[34]:


toronto_grouped.shape


# # Print each neighborhood along with the top 5 most common venues

# In[35]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# # Let's put that into a pandas dataframe
# First, let's write a function to sort the venues in descending order.

# In[36]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[37]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# # Cluster Neighborhoods 

# Run k-means to cluster the neighborhood into 5 clusters.

# In[38]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[ ]:





# In[39]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# Let's Change Cluster Labels into integer

# In[40]:


toronto_merged['Cluster Labels'].unique()


# In[41]:


toronto_merged['Cluster Labels'] = toronto_merged['Cluster Labels'].fillna(0)


# In[42]:


toronto_merged['Cluster Labels'] = toronto_merged['Cluster Labels'].astype(int)
toronto_merged['Cluster Labels'].unique()


# # Let's visualize resulting dataframe

# In[43]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




