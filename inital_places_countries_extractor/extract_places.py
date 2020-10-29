import re
import json
import string
import geograpy
import pycountry
import pandas as pd
from geograpy.extraction import Extractor

chars = re.escape(string.punctuation)
locations = pd.read_csv('locations.csv')
worldcities = pd.read_csv('worldcities.csv').sort_values('population', ascending=False)

pc = {}

def main():
	for i, location in locations.head(1000).iterrows():
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
		return e .find_entities()
	except:
		return []

def get_country(place):
	print(place)
	country = ""

	# FIRST LEVEL
	if len(worldcities[worldcities['city'] == place]) > 0:
		country = worldcities[worldcities['city'] == place]['country'].unique()[0]
	elif len(worldcities[worldcities['city_ascii'] == place]) > 0:
		country = worldcities[worldcities['city_ascii'] == place]['country'].unique()[0]
	if country:
		if pycountry.countries.get(name=country):
			return pycountry.countries.get(name=country).alpha_2
		else:
			return pycountry.countries.search_fuzzy(country)[0].alpha_2 

	# SECOND LEVEL
	if geograpy.locateCity(place):
		return geograpy.locateCity(place).country.iso

	# THIRD LEVEL
	countries = []
	t_country = ""
	places = place.split(" ")
	for temp_place in places:
		if len(worldcities[worldcities['city'] == temp_place]) > 0:
			t_country = worldcities[worldcities['city'] == temp_place]['country'].unique()[0]
		elif len(worldcities[worldcities['city_ascii'] == temp_place]) > 0:
			t_country = worldcities[worldcities['city_ascii'] == temp_place]['country'].unique()[0]
		if t_country:
			if pycountry.countries.get(name=t_country):
				countries.append(pycountry.countries.get(name=t_country).alpha_2)
			else:
				countries.append(pycountry.countries.search_fuzzy(t_country)[0].alpha_2)
	return '|'.join(list(set(countries)))



if __name__ == "__main__":
	main()