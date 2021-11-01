import pandas as pd
import numpy as np
from ast import literal_eval
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


train_url = 'http://courses.compute.dtu.dk/02807/2021/projects/project1/train.csv'
def f(x):
    try:
        return literal_eval(str(x))
    except Exception as e:
        return np.nan

def load_movies_data(url):
    df = pd.read_csv(url, dtype='unicode', low_memory=False)

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df.belongs_to_collection = df.belongs_to_collection.fillna('{}').apply(lambda x: f(x))
    df.genres = df.genres.fillna('[]').apply(lambda x: f(x))
    df.production_companies = df.production_companies.fillna('[]').apply(lambda x: f(x))
    df.production_countries = df.production_countries.fillna('[]').apply(lambda x: f(x))
    df.spoken_languages = df.spoken_languages.apply(lambda x: f(x))
    df.Keywords = df.Keywords.apply(lambda x: f(x))
    df.cast = df.cast.fillna('[]').apply(lambda x: f(x))
    df.crew = df.crew.fillna('[]').apply(lambda x: f(x))
    return df

train = load_movies_data(train_url)

for i, v in enumerate(train.genres.head()):
    print(i, v)

def genres_count(df):
    count = len(df['genres'])
    if count!=0:
        train['genres_count']=count
        return count
    else:
        train['genres_count']=0
        return 0


train.apply(genres_count)

fig = plt.figure()
ax = plt.axes()

ax.bar(train['genres_count'].value_counts().index, train['genres_count'].value_counts(), color = sns.color_palette())
fig.set_size_inches(8,5)

def binary_genre (data):
    data.genres=data.genres['name']
    if "Action" in data.genres:
        data['genres_Action'] = 1
    else:
        data['genres_Action'] = 0
    if "Comedy" in data.genres:
        data['genres_Comedy'] = 1
    else:
        data['genres_Comedy'] = 0
    if "Thriller" in data.genres:
        data['genres_Thriller'] = 1
    else:
        data['genres_Thriller'] = 0
    if "Drama" in data.genres:
        data['genres_Drama'] = 1
    else:
        data['genres_Drama'] = 0

train.apply(binary_genre)

