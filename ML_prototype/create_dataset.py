import pandas as pd

df_main = pd.read_csv('metabase_export.csv').dropna()
df_location = pd.read_csv('places_dataset/locations_with_places_and_countries.csv')

df_join = pd.merge(df_main, df_location, on ='location', how ='left')

df_join = df_join[['company', 'createdat', 'days', 'is_it_job', 'seniority', 'country']]

df_join.dropna(inplace=True)

df_join = df_join[df_join['days'] > 0]

# print(pd.pivot_table(df_join, values='company', index=['days'], aggfunc='count'))