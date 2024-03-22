#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cars_df = pd.read_csv('../Data/cars.csv')

cars_df_copy = cars_df.copy()

cars_df_copy['horsepower'] = pd.to_numeric(cars_df_copy['horsepower'], errors='coerce')

mean_horsepower = round(cars_df_copy['horsepower'].mean(), 1)

cars_df_copy['horsepower'].fillna(mean_horsepower, inplace=True)

cars_df_copy.to_csv('../Data/cars_cleaned.csv', index=False)

print(cars_df_copy)

#%%