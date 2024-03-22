#%%
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Create a OneHotEncoder instance
encoder = OneHotEncoder()

# Fit the encoder to the 'origin' column and transform it
encoded_origin = encoder.fit_transform(df[['origin']])

# Convert the encoded origin column into a DataFrame
encoded_origin_df = pd.DataFrame(encoded_origin.toarray(), columns=encoder.get_feature_names_out(['origin']))

# Concatenate the original DataFrame 'df' with the encoded origin DataFrame
df = pd.concat([df, encoded_origin_df], axis=1)

# Drop the original 'origin' column
df.drop(columns=['origin'], inplace=True)

#%%