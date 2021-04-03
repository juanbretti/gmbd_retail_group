# %%
## Import libraries ----
import numpy as np
from numpy.core.fromnumeric import product, sort
from numpy.lib.function_base import average
import pandas as pd
from pandas.core.common import flatten
from gensim.models import Word2Vec

import nltk
nltk.download('wordnet')

from sys import prefix
from tqdm import tqdm
import zipfile as zp

import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

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
products['products_mod'] = products['products_mod'].str.replace('\W', ' ')
# Split products into terms: Tokenize.
products['products_mod'] = products['products_mod'].str.split()

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
recipes_ingredients_list['ingredients'] = recipes_ingredients_list['ingredients'].str.replace('\W', ' ')

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
products_append = products_append.append(departments['department']).append(aisles['aisle'])
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
departments['vectors'] = w2v_applied(departments, 'department', 'department_id')
aisles['vectors'] = w2v_applied(aisles, 'aisle', 'aisle_id')

# %%
## Aggregate list to average ----

### `orders lists` ----
orders_filter = order_products_prior[order_products_prior.order_id < SPEED_LIMIT_ORDERS]
orders_filter = pd.merge(orders_filter, orders, on='order_id')
# Replace with the products
orders_filter_user = orders_filter.groupby(['order_id', 'user_id']).agg({'product_id':lambda x: list(set(x))})
orders_filter_user = orders_filter_user.reset_index()
orders_filter_user['vectors'] = orders_filter_user['product_id'].apply(lambda row: [products.loc[products['product_id']==item, 'vectors'].to_list()[0] for item in row])
orders_filter_user['vectors'] = orders_filter_user['vectors'].apply(lambda x: tuple(np.average(x, axis=0)))
# Calculate mean `reordered`
orders_filter_reordered = orders_filter.groupby(['user_id', 'product_id']).agg({'reordered': np.average})
# orders_filter_reordered['reordered'].value_counts()

### `users lists` ----
user_baskets = orders_filter_user.groupby('user_id')['vectors'].apply(list)
user_baskets_average = user_baskets.apply(lambda x: tuple(np.average(x, axis=0)))

# %%
## Add product and user vectors ----
def vector_to_df(df, id, vectors, name, vector_size=VECTOR_SIZE):
    temp = df.loc[:, [id, vectors]]
    return pd.DataFrame(temp[vectors].tolist(), index=temp[id], columns=[f'{name}_vector_{x}' for x in range(0,vector_size)]).reset_index()

user_vector = vector_to_df(user_baskets_average.reset_index(), 'user_id', 'vectors', 'user')
product_vector = vector_to_df(products, 'product_id', 'vectors', 'product')
aisle_vector = vector_to_df(aisles, 'aisle_id', 'vectors', 'aisle')
department_vector = vector_to_df(departments, 'department_id', 'vectors', 'department')

# product_vector = pd.merge(product_vector, products[['product_id', 'aisle_id', 'department_id']], on='product_id')
# product_vector = pd.merge(product_vector, aisle_vector, on='aisle_id')
# product_vector = pd.merge(product_vector, department_vector, on='department_id')

order_merged = orders_filter_reordered.reset_index()
order_merged = pd.merge(order_merged, user_vector, on='user_id')
order_merged = pd.merge(order_merged, product_vector, on='product_id')

# %%
## Split the data ---
# columns_to_exclude = ['product_id', 'user_id', 'aisle_id', 'department_id']
columns_to_exclude = ['product_id', 'user_id']
data = order_merged.drop(columns_to_exclude, axis=1)

### Split dataset ----
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['reordered'])
# Train
X_train = data_train.drop('reordered', axis=1)
y_train = data_train['reordered']
# Test
X_test = data_test.drop('reordered', axis=1)
y_test = data_test['reordered']

# %%
def timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# %%
## Model training ----
###  Hyperparameter tunning XGBoost, bayesian search ----
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn
# https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
# https://neptune.ai/blog/scikit-optimize
# C:\Users\juanb\OneDrive\GMBD\2020-06-29 - TERM 2\MACHINE LEARNING II (MBD-EN-BL2020J-1_32R202_380379)\Group assignment\!Delivery\ml_ii_group_f.ipynb

params = {
    'max_depth': list(range(3,10,2)),
    'min_child_weight': list(range(1,6,2)),
    'gamma': [i/10.0 for i in range(0,5)],
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

folds = 3
param_comb = 10
cv_ = 3

# Build the model
xgb_model_bayes = xgb.XGBRegressor(
    learning_rate = 0.3,
    n_estimators=10,
    max_depth=5,
    objective ='reg:squarederror',
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist', gpu_id=0, nthread=-1,
    random_state = 42)

# Build the search space
xgb_model_search_bayes = BayesSearchCV(xgb_model_bayes, search_spaces=params, n_iter=param_comb, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv_, verbose=3, random_state=42)

# Start the grid search
start_time = timer(None) # timing starts from this point for "start_time" variable
xgb_model_search_bayes.fit(X_train, y_train)
timer(start_time) # timing ends here for "start_time" variable

# %%
### Performance ----
# Checking the performance of the best model
# https://scikit-learn.org/stable/modules/model_evaluation.html

xgb_model_after_bayes_search = xgb_model_search_bayes.best_estimator_
xgb_scores_bayes_tunned = cross_val_score(xgb_model_after_bayes_search, X_test, y_test, scoring='neg_mean_squared_error', cv=10)
print("neg_mean_squared_error: %0.4f (+/- %0.2f)" % (np.median(xgb_scores_bayes_tunned), np.std(xgb_scores_bayes_tunned)))

# %%
### Feature importance ----
plt.rcParams['figure.figsize'] = [15, 15]
xgb_model_bayes.fit(X_train,y_train)
xgb.plot_importance(xgb_model_bayes)

# %%
## Probability of a specific user buying a recipe ----
### Buy predicting each product individually ----

# Reformat for the prediction
ingredients_vector = vector_to_df(recipes_ingredients, 'ingredient_id', 'vectors', 'product')

def probability_each_product_individually(user_test):
    # Cross join of `user` and `ingredients`
    # https://stackoverflow.com/a/48255115/3780957
    user_test_ingredient_cross_join = user_vector[user_vector['user_id']==user_test].assign(foo=1).merge(ingredients_vector.assign(foo=1)).drop('foo', 1)
    # Prepare data and predict
    columns_to_exclude = ['ingredient_id', 'user_id']
    data = user_test_ingredient_cross_join.drop(columns_to_exclude, axis=1)
    # Final prediction
    user_test_predict = user_test_ingredient_cross_join[['ingredient_id', 'user_id']].copy()
    user_test_predict['probability_reorder'] = xgb_model_bayes.predict(data)
    # Merge original recipes with the user probability of `reorder`
    user_test_recipes = pd.merge(recipes_ingredients_list, user_test_predict, on='ingredient_id')
    # List of recipes sorted by average probability
    return user_test_recipes.groupby(['name', 'recipes_id'])['probability_reorder'].mean().sort_values(ascending=False)

probability_each_product_individually(204484)
# %%
### By predicting the whole recipe ----

# Reformat for the prediction
recipes_ingredients_vector = pd.merge(recipes_ingredients_list, recipes_ingredients, on='ingredient_id')
recipes_ingredients_vector = recipes_ingredients_vector.groupby(['name', 'recipes_id'])['vectors'].apply(list)
recipes_ingredients_vector_ = recipes_ingredients_vector.apply(lambda x: tuple(np.average(x, axis=0)))
recipes_ingredients_vector_ = vector_to_df(recipes_ingredients_vector_.reset_index(), 'recipes_id', 'vectors', 'product')

def probability_whole_recipe(user_test):
    # Cross join of `user` and `ingredients`
    user_test_ingredient_cross_join = user_vector[user_vector['user_id']==user_test].assign(foo=1).merge(recipes_ingredients_vector_.assign(foo=1)).drop('foo', 1)
    # Prepare data and predict
    columns_to_exclude = ['recipes_id', 'user_id']
    data = user_test_ingredient_cross_join.drop(columns_to_exclude, axis=1)
    # Final prediction
    user_test_predict = user_test_ingredient_cross_join[['recipes_id', 'user_id']].copy()
    user_test_predict['probability_reorder'] = xgb_model_bayes.predict(data)
    # Merge original recipes with the user probability of `reorder`
    user_test_recipes = pd.merge(recipes_ingredients_vector.reset_index(), user_test_predict, on='recipes_id')
    # List of recipes sorted by average probability
    return user_test_recipes.groupby(['name', 'recipes_id'])['probability_reorder'].mean().sort_values(ascending=False)

probability_whole_recipe(204484)
# %%