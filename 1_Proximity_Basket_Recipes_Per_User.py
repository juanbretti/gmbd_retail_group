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

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.express as px
# %%
## Constants ----
# Limit of number of orders to process
SPEED_LIMIT_ORDERS = 1000
# Limit of number of recipes to process
SPEED_LIMIT_RECIPES = 3000
# Number of orders/baskets to pull similar to the requested
NUMBER_OF_RELATED_RECIPES = 5
# Number of dimensions of the vector annoy is going to store. 
VECTOR_SIZE = 30
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
products['products_mod'] = products['products_mod'].str.replace('\W', ' ', regex=True)
# Split products into terms: Tokenize.
products['products_mod'] = products['products_mod'].str.split()

# Merge the department and aisle names into the dataframe. 
products = pd.merge(products, departments, on="department_id", how='outer')
products = pd.merge(products, aisles, on="aisle_id", how='outer')

# Remove products that are not food
no_food_department = ['personal care', 'household', 'babies', 'pets']
products = products[~products['department'].isin(no_food_department)]

# https://stackoverflow.com/a/43898233/3780957
# https://stackoverflow.com/a/57225427/3780957
products['products_mod'] = products[['products_mod', 'aisle', 'department']].values.tolist()
#products['products_mod'] = products['products_mod'].values.tolist()
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
recipes_ingredients_raw = recipes.head(SPEED_LIMIT_RECIPES).copy()
recipes_ingredients_raw['recipes_id'] = recipes_ingredients_raw.index
recipes_ingredients_raw['ingredients'] = recipes_ingredients_raw['ingredients'].apply(lambda x: strToList(x, ','))

recipes_ingredients_list = recipes_ingredients_raw.explode('ingredients')
recipes_ingredients_list['ingredients'] = recipes_ingredients_list['ingredients'].str.lower()
recipes_ingredients_list['ingredients'] = recipes_ingredients_list['ingredients'].str.replace('\W', ' ', regex=True)

# Dataframe of unique values
recipes_ingredients = pd.DataFrame(recipes_ingredients_list['ingredients'].unique(), columns=['ingredient_name'])
recipes_ingredients['ingredient_id'] = recipes_ingredients.index

# Stem and lemma
recipes_ingredients['ingredients_lemma'] = recipes_ingredients['ingredient_name'].str.split()
recipes_ingredients['ingredients_lemma'] = recipes_ingredients['ingredients_lemma'].apply(lambda row:[lemma.lemmatize(item) for item in row])
recipes_ingredients['ingredients_lemma'] = recipes_ingredients['ingredients_lemma'].apply(lambda row:[sno.stem(item) for item in row])

# Recipes list with the `ingredient_id`
recipes_ingredients_list = pd.merge(recipes_ingredients_list, recipes_ingredients, left_on='ingredients', right_on='ingredient_name')

# %%
## `Word2Vec` model ----
### Train the model ----

# The `products_lemma` column is ready to be used as an input for the `Word2Vec` model. 
products_append = products['products_lemma'].append(recipes_ingredients['ingredients_lemma'])
# Remove duplicates. Alternative: https://stackoverflow.com/a/3724558/3780957
products_append = products_append.apply(set).apply(list)
# to define the maximun window
window_max = max(products_append.apply(lambda x:len(x)))
# Create the model itself
w2vec_model = Word2Vec(list(products_append), size=VECTOR_SIZE, window=window_max, min_count=1, workers=-1)

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
recipes_ingredients['vectors'] = w2v_applied(recipes_ingredients, 'ingredients_lemma', 'ingredient_id')

# %%
## Train `annoy` ----
# pv: product vectors
# iv: ingredient vectors
# bl: basket list
# rl: recipe list

def annoy_build(df, id, metric='euclidean'):
    m = AnnoyIndex(VECTOR_SIZE, metric=metric) 
    m.set_seed(42)
    for _, row in df.iterrows():
        m.add_item(row[id], row['vectors'])
    m.build(TREE_QUERIES)
    return m

### `product` ----
pv = annoy_build(products, 'product_id')

### `ingredients` ----
iv = annoy_build(recipes_ingredients, 'ingredient_id')

# %%
### `orders lists` ----
# Limit the number of orders
orders_filter = order_products_prior[order_products_prior.order_id < SPEED_LIMIT_ORDERS]
# Remove products that are not food
orders_filter = orders_filter[orders_filter['product_id'].isin(products['product_id'])]
orders_filter = pd.merge(orders_filter, orders, on='order_id')
order_baskets = orders_filter.groupby('user_id')['product_id'].apply(set).apply(list)

order_w2v = dict()
for index, row in tqdm(order_baskets.items()):
    word_vector = list()
    for item_id in row:
        word_vector.append(pv.get_item_vector(item_id))
    order_w2v[index] = np.average(word_vector, axis=0)
df_order_baskets = pd.DataFrame(order_baskets.items(), columns=['user_id', 'product_id'])
df_order_baskets['vectors'] = order_w2v.values()

bl = annoy_build(df_order_baskets, 'user_id')

### `recipes lists` ----
recipes_list = recipes_ingredients_list.groupby('recipes_id')['ingredient_id'].apply(list)

list_w2v = dict()
for index, row in tqdm(recipes_list.items()):
    word_vector = list()
    for item_id in row:
        word_vector.append(iv.get_item_vector(item_id))
    list_w2v[index] = np.average(word_vector, axis=0)
df_recipes_list = pd.DataFrame(recipes_list.items(), columns=['recipes_id', 'ingredient_id'])
df_recipes_list['vectors'] = list_w2v.values()

rl = annoy_build(df_recipes_list, 'recipes_id')

# %%
## TSNE plot ----
def tsne_plot(df, title='Comparison of datasets', auto_open=True, id='id'):

    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=42)
    tsne_values = tsne_model.fit_transform(list(df['vectors']))

    df['tsne-2d-one'] = tsne_values[:, 0]
    df['tsne-2d-two'] = tsne_values[:, 1]
    df['hover'] = 'ID: ' + df[id].astype(str)
    
    df.sort_values(by='source', ascending=False, inplace=True)

    fig = px.scatter(df, x="tsne-2d-one", y="tsne-2d-two",
                    color='source', 
                    title=title,
                    hover_data=['hover'],
                    labels={
                        "tsne-2d-one": "Dimension one",
                        "tsne-2d-two": "Dimension two",
                        "source": "Source reference"
                    })
    pyo.plot(fig, filename=f'plots/tsne_plot_{title}.html', auto_open=auto_open)

# %%
### Orders vs recipes ----
df_r = df_recipes_list[['recipes_id', 'vectors']].rename(columns={'recipes_id': 'id'}).assign(source='Recipe')
df_o = df_order_baskets[['user_id', 'vectors']].rename(columns={'user_id': 'id'}).assign(source='Order')
df_ro = pd.concat([df_r, df_o])

tsne_plot(df=df_ro, title='Comparison between `Recipes` and `Orders`')

# %%
### Products vs ingredients ----
df_r = recipes_ingredients[['ingredient_id', 'ingredient_name', 'vectors']].\
    rename(columns={'ingredient_id': 'id', 'ingredient_name': 'name'}).\
    assign(source='Recipe')
df_o = products[['product_id', 'product_name', 'vectors']].\
    rename(columns={'product_id': 'id', 'product_name': 'name'}).\
    assign(source='Order')
df_ro = pd.concat([df_r, df_o])

tsne_plot(df=df_ro.sample(n=2000, random_state=42), 
    title='Comparison between `Recipes ingredients` and `Orders products`',
    id='name')

# %%
## Closest Recipes (by ticket) ----
### Search for recommendations ----
order_related_recipes = pd.DataFrame()
for index, row in tqdm(df_order_baskets.iterrows()):
    row['recipes_related'] = rl.get_nns_by_vector(row['vectors'], NUMBER_OF_RELATED_RECIPES, search_k=-1, include_distances=False)
    order_related_recipes = order_related_recipes.append(row)

# %%
### Convert `id` to `name` ----
# Add product information
order_related_recipes_p = order_related_recipes
order_related_recipes_p = order_related_recipes_p.explode('product_id')
order_related_recipes_p = pd.merge(order_related_recipes_p, products[['product_id', 'product_name']], left_on='product_id', right_on='product_id')
order_related_recipes_p = order_related_recipes_p.groupby('user_id')['product_name'].apply(list)

# Add recipes information
order_related_recipes_r = order_related_recipes
order_related_recipes_r = order_related_recipes_r.explode('recipes_related')
order_related_recipes_r = pd.merge(order_related_recipes_r, recipes_ingredients_raw[['recipes_id', 'name']], left_on='recipes_related', right_on='recipes_id')
order_related_recipes_r = order_related_recipes_r.groupby('user_id')['name'].apply(list)

# Merge the two previous
order_related_recipes_list = order_related_recipes
order_related_recipes_list = pd.merge(order_related_recipes_list, order_related_recipes_p, on='user_id')
order_related_recipes_list = pd.merge(order_related_recipes_list, order_related_recipes_r, on='user_id')
order_related_recipes_list.reset_index(inplace=True)

# %%
### Print test of the model ----
def print_test(x):
    order_info = order_related_recipes_list.loc[x, ['user_id', 'product_name', 'name']]
    print('** User info **')
    print(order_info)
    print('\n')
    print('** Products purchased by user **')
    print(order_info['product_name'])
    print('\n')
    print('** First recommended recipes **')
    print(order_info['name'][:3])

print_test(100)
# %%
