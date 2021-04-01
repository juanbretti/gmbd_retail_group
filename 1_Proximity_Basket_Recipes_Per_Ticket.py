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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %%
## Constants ----
SPEED_LIMIT_ORDERS = 10000
SPEED_LIMIT_RECIPES = 10000
# Number of orders/baskets to pull similar to the requested
NUMBER_OF_RELATED_RECIPES = 5
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
## Data load ----
### Read CSV ----
# Data column description
# https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b
# https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv
# https://www.kaggle.com/c/instacart-market-basket-analysis

zf1 = zp.ZipFile("data/order_products__prior.csv.zip")
order_products_prior = pd.read_csv(zf1.open('order_products__prior.csv'))

zf1 = zp.ZipFile("data/orders.csv.zip")
orders = pd.read_csv(zf1.open('orders.csv'))

zf1 = zp.ZipFile("data/products.csv.zip")
products = pd.read_csv(zf1.open('products.csv'))

zf1 = zp.ZipFile("data/departments.csv.zip")
departments = pd.read_csv(zf1.open('departments.csv'))

zf1 = zp.ZipFile("data/aisles.csv.zip")
aisles = pd.read_csv(zf1.open('aisles.csv'))

zf1 = zp.ZipFile("data/RAW_recipes.csv.zip")
recipes = pd.read_csv(zf1.open('RAW_recipes.csv'))

# %%
### Products ----

# Make everything lowercase.
products['products_mod'] = products['product_name'].str.lower()
# Clean special characters.
products['products_mod'] = products['products_mod'].str.replace('\W', ' ')
# Split products into terms: Tokenize.
products['products_mod'] = products['products_mod'].str.split()

# Merge the department and aisle names into the dataframe. 
products = pd.merge(products, departments, on="department_id", how='outer')
products = pd.merge(products, aisles, on="aisle_id", how='outer')

# https://stackoverflow.com/a/43898233/3780957
# https://stackoverflow.com/a/57225427/3780957
# Remove synonyms here in the list
products['products_mod'] = products[['products_mod', 'aisle', 'department']].values.tolist()
products['products_mod'] = products['products_mod'].apply(lambda x:list(flatten(x)))

# %%
# Steam and lemmatisation of the product name
# https://stackoverflow.com/a/24663617/3780957
# https://stackoverflow.com/a/25082458/3780957
# https://en.wikipedia.org/wiki/Lemmatisation

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')

products['products_lemma'] = products['products_mod'].apply(lambda row:[lemma.lemmatize(item) for item in row])
products['products_lemma'] = products['products_lemma'].apply(lambda row:[sno.stem(item) for item in row])

# %%
### Recipes ingredients ----
#### Format conversion function ----
# https://www.kaggle.com/zinayida/data-exploration-feature-engineering-pipeline
def strToList(list_l, splitSymbol):
    list_l = list_l.split(splitSymbol)
    temp = list()
    for l in list_l: 
        l = l.replace("[",'').replace("]",'').replace("'", '').replace("\"", '').strip()
        temp.append(l)
    return temp

# %%
#### Conversion ----
recipes_products_list = recipes.head(SPEED_LIMIT_RECIPES).copy()
recipes_products_list['recipes_id'] = recipes_products_list.index
recipes_products_list['ingredients'] = recipes_products_list['ingredients'].apply(lambda x: strToList(x, ','))
recipes_products_list = recipes_products_list.explode('ingredients')
recipes_products_list['ingredients'] = recipes_products_list['ingredients'].str.lower()
recipes_products_list['ingredients'] = recipes_products_list['ingredients'].str.replace('\W', ' ')

# Dataframe of unique values
recipes_products = pd.DataFrame(recipes_products_list['ingredients'].unique(), columns=['ingredient_name'])
recipes_products['ingredient_id'] = recipes_products.index

# Stem and lemma
recipes_products['ingredients_lemma'] = recipes_products['ingredient_name'].str.split()
recipes_products['ingredients_lemma'] = recipes_products['ingredients_lemma'].apply(lambda row:[lemma.lemmatize(item) for item in row])
recipes_products['ingredients_lemma'] = recipes_products['ingredients_lemma'].apply(lambda row:[sno.stem(item) for item in row])

# Recipes list with the `ingredient_id`
recipes_products_list = recipes_products_list.merge(recipes_products, left_on='ingredients', right_on='ingredient_name')

# %%
## `Word2Vec` model ----
### Train the model ----

# The `products_lemma` column is ready to be used as an input for the `Word2Vec` model. 
products_append = products['products_lemma'].append(recipes_products['ingredients_lemma'])
# https://stackoverflow.com/a/3724558/3780957
products_append = pd.Series([list(x) for x in set(tuple(x) for x in products_append)])
# to define the maximun window
window_max = max(products_append.apply(lambda x:len(x)))
# Create the model itself
w2vec_model = Word2Vec(list(products_append), size=20, window=window_max, min_count=1, workers=-1)

# %%
### Vector calculation for products ----
def w2v_applied(df, prod, id):
    prods_w2v = dict()
    for _, product in tqdm(df.iterrows()):
        word_vector = list()
        for word in product[prod]:
            word_vector.append(w2vec_model.wv[word])
        prods_w2v[product[id]] = np.average(word_vector, axis=0)
    return prods_w2v.values()

products['vectors'] = w2v_applied(products, 'products_lemma', 'product_id')
recipes_products['vectors'] = w2v_applied(recipes_products, 'ingredients_lemma', 'ingredient_id')

# %%
## Train `annoy` ----
# pv: product vectors
# iv: ingredient vectors
# bl: basket list
# rl: recipe list

### `product` ----
pv = AnnoyIndex(VECTOR_SIZE, metric='manhattan') 
pv.set_seed(42)
for index, row in products.iterrows():
    pv.add_item(row['product_id'], row['vectors'])
pv.build(TREE_QUERIES)

### `ingredients` ----
iv = AnnoyIndex(VECTOR_SIZE, metric='manhattan') 
iv.set_seed(42)
for index, row in recipes_products.iterrows():
    iv.add_item(row['ingredient_id'], row['vectors'])
iv.build(TREE_QUERIES)

# %%
### `orders lists` ----
orders_filter = order_products_prior[order_products_prior.order_id < SPEED_LIMIT_ORDERS]
order_baskets = orders_filter.groupby('order_id')['product_id'].apply(list)

order_w2v = dict()
for index, row in tqdm(order_baskets.items()):
    word_vector = list()
    for item_id in row:
        word_vector.append(pv.get_item_vector(item_id))
    order_w2v[index] = np.average(word_vector, axis=0)
df_order_baskets = pd.DataFrame(order_baskets.items(), columns=['order_id', 'product_id'])
df_order_baskets['vectors'] = order_w2v.values()

bl = AnnoyIndex(VECTOR_SIZE, metric='manhattan')
bl.set_seed(42)
for index, row in df_order_baskets.iterrows():
    bl.add_item(row['order_id'], row['vectors'])
bl.build(TREE_QUERIES)

### `recipes lists` ----
recipes_list = recipes_products_list.groupby('recipes_id')['ingredient_id'].apply(list)

list_w2v = dict()
for index, row in tqdm(recipes_list.items()):
    word_vector = list()
    for item_id in row:
        word_vector.append(iv.get_item_vector(item_id))
    list_w2v[index] = np.average(word_vector, axis=0)
df_recipes_list = pd.DataFrame(recipes_list.items(), columns=['recipes_id', 'ingredient_id'])
df_recipes_list['vectors'] = list_w2v.values()

rl = AnnoyIndex(VECTOR_SIZE, metric='manhattan')
rl.set_seed(42)
for index, row in df_recipes_list.iterrows():
    rl.add_item(row['recipes_id'], row['vectors'])
rl.build(TREE_QUERIES)

# %%
## Closest Recipes (by ticket) ----

order_related_recipts = pd.DataFrame()
for index, row in tqdm(df_order_baskets.iterrows()):
    row['recipts_related'] = rl.get_nns_by_vector(row['vectors'], NUMBER_OF_RELATED_RECIPES, search_k=-1, include_distances=False)
    order_related_recipts = order_related_recipts.append(row)

# %%
