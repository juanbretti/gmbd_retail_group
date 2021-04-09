# %%
## Import libraries ----
import numpy as np
import pandas as pd

import zipfile as zp
import pickle
from art import *

import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import plotly.express as px

# %%
## Constants ----
# Variables save location
SAVED_PKL = 'store/saved_variables.pkl'
# Color constants for the console
COLOR_CONSTANT = {'input': '\033[94m', 'warning': '\033[93m', 'error': '\033[91m', 'note': '\033[96m', 'end': '\033[0m'}
# Number of orders/baskets to pull similar to the requested
NUMBER_OF_RETURNS = 10

# %%
# Load the model from disk
xgb_model_bayes, recipes_ingredients, user_merged, recipes_ingredients_list, VECTOR_SIZE, user_baskets_average = pickle.load(open(SAVED_PKL, 'rb'))

# %%
## Add product and user vectors ----
def vector_to_df(df, id, vectors, name, vector_size=VECTOR_SIZE):
    temp = df.loc[:, [id, vectors]]
    return pd.DataFrame(temp[vectors].tolist(), index=temp[id], columns=[f'{name}_vector_{x}' for x in range(0,vector_size)]).reset_index()

# %%
## Probability of a specific user buying a recipe ----
### Buy predicting each product individually ----

# Reformat for the prediction
ingredients_vector = vector_to_df(recipes_ingredients, 'ingredient_id', 'vectors', 'product')

def probability_each_product_individually(user_test, user_merged=user_merged):
    # Cross join of `user` and `ingredients`
    # https://stackoverflow.com/a/48255115/3780957
    user_test_ingredient_cross_join = user_merged[user_merged['user_id']==user_test].assign(foo=1).merge(ingredients_vector.assign(foo=1)).drop('foo', 1)
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

# %%
### By predicting the whole recipe ----

# Reformat for the prediction
recipes_ingredients_vector = pd.merge(recipes_ingredients_list, recipes_ingredients, on='ingredient_id')
recipes_ingredients_vector = recipes_ingredients_vector.groupby(['name', 'recipes_id'])['vectors'].apply(list)
recipes_ingredients_vector_ = recipes_ingredients_vector.apply(lambda x: tuple(np.average(x, axis=0)))
recipes_ingredients_vector_ = vector_to_df(recipes_ingredients_vector_.reset_index(), 'recipes_id', 'vectors', 'product')

def probability_whole_recipe(user_test, user_merged=user_merged):
    # Cross join of `user` and `ingredients`
    user_test_ingredient_cross_join = user_merged[user_merged['user_id']==user_test].assign(foo=1).merge(recipes_ingredients_vector_.assign(foo=1)).drop('foo', 1)
    # Prepare data and predict
    columns_to_exclude = ['recipes_id', 'user_id']
    data = user_test_ingredient_cross_join.drop(columns_to_exclude, axis=1)
    # Final prediction
    user_test_predict = user_test_ingredient_cross_join[['recipes_id', 'user_id']].copy()
    user_test_predict['probability_reorder'] = xgb_model_bayes.predict(data)
    # Merge original recipes with the user probability of `reorder`
    user_test_recipes = pd.merge(recipes_ingredients_vector.reset_index(), user_test_predict, on='recipes_id')
    # List of recipes sorted by average probability
    return user_test_recipes[['name', 'recipes_id', 'probability_reorder']].sort_values(by='probability_reorder', ascending=False)

# %%
## TSNE model plot function, with selection ----

def tsne_plot(selection, df=user_baskets_average, title='Selected `user` between others', auto_open=True):

    # Data sample, to speedup the execution
    df_tsne_data = df.reset_index()
    df_tsne_data['size'] = 1
    df_tsne_data['color'] = 'Others'

    df_tsne_data.loc[df_tsne_data['user_id'] == selection, 'size'] = 5
    df_tsne_data.loc[df_tsne_data['user_id'] == selection, 'color'] = 'Selection'

    # Train the TSNE MODEL
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=42)
    tsne_values = tsne_model.fit_transform(list(df_tsne_data['vectors']))

    df_tsne_data['tsne-2d-one'] = tsne_values[:, 0]
    df_tsne_data['tsne-2d-two'] = tsne_values[:, 1]
    df_tsne_data['hover'] = 'User ID: ' + df_tsne_data['user_id'].astype(str)
    
    df_tsne_data.sort_values(by='color', ascending=False, inplace=True)

    fig = px.scatter(df_tsne_data, x="tsne-2d-one", y="tsne-2d-two",
                    color='color', 
                    size="size", size_max=8,
                    title=title,
                    hover_data=['hover'],
                    labels={
                        "tsne-2d-one": "Dimension one",
                        "tsne-2d-two": "Dimension two",
                        "color": "Color reference"
                    })
    pyo.plot(fig, filename=f'plots/tsne_plot_{title}.html', auto_open=auto_open)

# %%
## Interface ----
### Interface auxiliary functions ----

# Clean the terminal console
def clear_console(n=100):
    print('\n'*n)
    return None

# Request the user an input
def read_positive(message_try='Type a value', message_error='Only accepts values like `42`, try again.', color='input'):
    # Loop until the user enters a value in the possibles
    correct_value = False
    while not correct_value:
        try:
            user_value = int(input(COLOR_CONSTANT[color] + message_try + COLOR_CONSTANT['end']))
            if user_value > 0:
                correct_value = True
            else:
                correct_value = False
                raise ValueError
        except ValueError:
            correct_value = False
            print(COLOR_CONSTANT['error'] + message_error + COLOR_CONSTANT['end'])
    return user_value

def print_color(txt, color='note'):
    print(COLOR_CONSTANT[color] + txt + COLOR_CONSTANT['end'])

# %%
### Run user interface ----
clear_console()
tprint("Market basket analysis\nby Group C","Standard")

INPUT_TYPES = {1: 'Available users', 2: 'Probability considering product individually', 3: 'Probability considering whole recipe', 4: 'Exit'}
select_continue = 1  # Start the loop

# Infinite loop, will stop  by pressing Ctrl+C or selecting 'Exit' when prompt
while select_continue < max(INPUT_TYPES.keys()):
    # Prompt the user to continue or stop the infinite rounds
    select_continue = read_positive(message_try='Choose your option %s\n>>: ' % (
                                        ' '.join('\n{}: {}'.format(k, v) for k, v in INPUT_TYPES.items())),
                                    message_error='Only accepts the values %s, try again.' % (tuple(INPUT_TYPES.keys()),)
                                    )

    if select_continue < max(INPUT_TYPES.keys()) :
        clear_console()
        tprint(INPUT_TYPES[select_continue].replace(' ', '\n'), "Standard")

    # Selection based on the key from `INPUT_TYPES`
    if select_continue == 1:
        print_color('Available users')
        print(' '.join(f'{x}' for x in user_merged['user_id'].sort_values()))

    elif select_continue == 2:
        select_ = read_positive(message_try='User ID [e.g., `10441`]: ')
        print_color('Recommended recipes:')
        print(probability_each_product_individually(select_)\
            .reset_index()\
            .rename(columns={"name": "Recipe name", "recipes_id": "ID", "probability_reorder": "Probability of reorder"})\
            .head(NUMBER_OF_RETURNS)\
            .to_string(index=False))
        print('\n')
        print_color('Plot will open in Internet browser, please wait...')
        print('\n')
        tsne_plot(select_)

    elif select_continue == 3:
        select_ = read_positive(message_try='User ID [e.g., `10441`]: ')
        print_color('Recommended recipes:')
        print(probability_whole_recipe(select_)\
            .rename(columns={"name": "Recipe name", "recipes_id": "ID", "probability_reorder": "Probability of reorder"})\
            .head(NUMBER_OF_RETURNS)\
            .to_string(index=False))
        print('\n')
        print_color('Plot will open in Internet browser, please wait...')
        print('\n')
        tsne_plot(select_)

# %%
