import re
import json
import string
import geograpy
import pycountry
import pandas as pd
from geograpy.extraction import Extractor

chars = re.escape(string.punctuation)
locations = pd.read_csv('locations.csv')
worldcities = pd.read_csv('worldcities.csv')

pc = {}

def main():
	for i, location in locations.head(10).iterrows():
		clean_location = re.sub(r"[^\w,]|_", ' ', location['location'])
		places = get_places(clean_location)
		countries = []
		for place in places:
			if place in pc:
				countries.append(pc[place])
			else:
				country = get_country(place)
				countries.append(country)
				pc[place] = country
		locations.loc[i,'places'] = "|".join(places)
		locations.loc[i,'countries'] = "|".join(list(set(countries)))
	locations.to_csv('locations_with_places_and_countries.csv')
	print(json.dumps(pc, indent=4, sort_keys=True))

# all_places = pd.DataFrame(list(set(all_places)), columns=['places'])
# all_countries = pd.DataFrame(list(set(all_countries)), columns=['countries'])
# pd.DataFrame(list(set(list_places)), columns=['places']).to_csv('all_places.csv')

def get_places(location):
	try:
		e = Extractor(text=location)
		return e.find_entities()
	except:
		return []

def get_country(place):
	country = ""
	if worldcities['city'].str.contains(place).any():
		print(place)
		print(worldcities[worldcities['city'] == place]['country'])
		country = worldcities[worldcities['city'] == place]['country'].unique()[0]
	else:
		if worldcities['city_ascii'].str.contains(place).any():
			country = worldcities[worldcities['city_ascii'] == place]['country'].unique()[0]
	if country == "":
		try:
			return geograpy.locateCity(place).country.iso
		except:
			return country
	else:
		return pycountry.countries.get(name=country).alpha_2

if __name__ == "__main__":
    main()