# %%
## Import libraries ----
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from annoy import AnnoyIndex
from gensim.models import Word2Vec

import nltk
nltk.download('wordnet')

from tqdm import tqdm
import zipfile as zp
from art import *
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.express as px

import random
import time

# %%
## Constants ----
ORDERS_SPEED_LIMIT = 10000
# Number of orders/baskets to pull similar to the requested
NUMBER_OF_RETURNS = 5
# Number of dimensions of the vector annoy is going to store. 
VECTOR_SIZE = 20
# Number of trees for queries. When making a query the more trees the easier it is to go down the right path. 
TREE_QUERIES = 10
# Number of product recommendation as maximum
NUMBER_OUTPUT_PRODUCTS = 10
# Sample size for the TSNE model and plot
TSNE_SAMPLE_SIZE = 1000
# Threshold for a minimum support
THRESHOLD_SUPPORT = 1
# Threshold for the maximun number of products to bring
THRESHOLD_TOP = 100

# %%
## Read data ----
# Data column description
# https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b
# https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv
# https://www.kaggle.com/c/instacart-market-basket-analysis

zf1 = zp.ZipFile("Data/order_products__prior.csv.zip")
order_products_prior = pd.read_csv(zf1.open('order_products__prior.csv'))

zf1 = zp.ZipFile("Data/orders.csv.zip")
orders = pd.read_csv(zf1.open('orders.csv'))

zf1 = zp.ZipFile("Data/products.csv.zip")
products = pd.read_csv(zf1.open('products.csv'))

zf1 = zp.ZipFile("Data/departments.csv.zip")
departments = pd.read_csv(zf1.open('departments.csv'))

zf1 = zp.ZipFile("Data/aisles.csv.zip")
aisles = pd.read_csv(zf1.open('aisles.csv'))

zf1 = zp.ZipFile("Data/RAW_recipes.csv.zip")
recipes = pd.read_csv(zf1.open('RAW_recipes.csv'))

# %%
### Synonyms ----
# Manually extracted from https://www.thesaurus.com/browse/synonym
departments_synonyms = pd.read_csv('Data/departments synonyms.csv')

# %%
### Merge data ----

# Make everything lowercase.
products['products_mod'] = products['product_name'].str.lower()
# Clean special characters.
products['products_mod'] = products['products_mod'].str.replace('\W', ' ')
# Split products into terms: Tokenize.
products['products_mod'] = products['products_mod'].str.split()

# Merge the synonyms perse
departments_synonyms = departments_synonyms.groupby('department')['synonyms'].apply(list)
departments_synonyms = pd.merge(departments, departments_synonyms, on="department", how='outer').fillna('')

# Merge the department and aisle names into the dataframe. 
products = pd.merge(products, departments_synonyms, on="department_id", how='outer')
products = pd.merge(products, aisles, on="aisle_id", how='outer')

# https://stackoverflow.com/a/43898233/3780957
# https://stackoverflow.com/a/57225427/3780957
# Remove synonyms here in the list
products['products_mod'] = products[['products_mod', 'aisle', 'department', 'synonyms']].values.tolist()
products['products_mod'] = products['products_mod'].apply(lambda x:list(flatten(x)))

# %%
# Lemmatisation of the product name
# https://stackoverflow.com/a/24663617/3780957
# https://stackoverflow.com/a/25082458/3780957
# https://en.wikipedia.org/wiki/Lemmatisation

lemma = nltk.wordnet.WordNetLemmatizer()
products['products_lemma'] = products['products_mod'].apply(lambda row:[lemma.lemmatize(item) for item in row])

# %%
## `Word2Vec` model ----
### Train the model ----
# The `products_lemma` column is ready to be used as an input for the `Word2Vec` model. 

# to define the maximun window
# max(products['products_lemma'].apply(lambda x:len(x)))

# size=20: In order to make `TSNE` a little bit quicker and for memory efficiency we're going to use 20 dimensions.
# window=49: In order to make sure all words are used in training the model, we're going to set a large.
# workers=-1: means all the available cores in the CPU.
w2vec_model = Word2Vec(list(products['products_lemma']), size=20, window=49, min_count=1, workers=-1)

# %%
### Create dictionary of words ----
prod_word = dict()
for w in w2vec_model.wv.vocab:
    prod_word[w] = w2vec_model[w]

# print(list(prod_word.items())[0])

# %%
### Vector calculation for products ----
# Loop through each product and obtain the average of each string that makes a product. <br>
# For this dictionary we're not gong to store the product name as its key, but the product ID. 

# Cycle through each word in the product name to generate the vector.
prods_w2v = dict()
for row, product in tqdm(products.iterrows()):
    word_vector = list()
    for word in product['products_lemma']:
        word_vector.append(prod_word[word])

    prods_w2v[product['product_id']] = np.average(word_vector, axis=0)

# print(list(prods_w2v.items())[0])

# Save vector values in list form to the dataframe.
products['vectors'] = prods_w2v.values()
