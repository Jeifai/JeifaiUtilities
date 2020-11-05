import pandas as pd

df_main = pd.read_csv('metabase_export.csv').dropna()

df_location = pd.read_csv('places_dataset/locations_with_places_and_countries.csv')

df_join = pd.merge(df_main, df_location, on ='location', how ='left')

df_join.dropna(inplace=True)

df_join = df_join[df_join['days'] > 0]

df_join.company = pd.Categorical(df_join.company)
df_join['company_code'] = df_join.company.cat.codes

df_join.country = pd.Categorical(df_join.country)
df_join['country_code'] = df_join.country.cat.codes

df_join = df_join[['day', 'month', 'days', 'is_it_job', 'seniority', 'company_code', 'country_code']]

print(df_join.head(5))

df_join.to_csv('ml_dataset.csv')

# print(pd.pivot_table(df_join, values='company', index=['days'], aggfunc='count'))