import re
import json
import requests
import pandas as pd
from collections import Counter

locations = pd.read_csv('locations.csv')
geonames = pd.read_csv('geonames.txt', sep="	", header=None, low_memory=False)
geonames.columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
				'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code', 'admin2_code ', 
				'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification_date']
geonames.sort_values('population', ascending=False, inplace=True)
countrynames = requests.get("https://raw.githubusercontent.com/mledoze/countries/master/dist/countries.json").json()

def main():
	for i, dict_location in locations.iterrows():
		print(dict_location['location'])
		location = dict_location['location']
		places = get_places(location)
		country_map = {}
		country_map['is_country'] = []
		country_map['others'] = []
		for place in places:
			get_country(country_map, place)
		country = pick_country(country_map)
		locations.loc[i,'country'] = country
		print("\t\t\t", country)
	locations.to_csv('locations_with_places_and_countries.csv')

def get_places(location):
	places = []
	try:
		for i in re.split(r'[^\w ]', location):
			temp_place = re.sub(r"[^\w,\w ]|_", ',', i.strip())
			if temp_place:
				places.append(temp_place)
		if len(places) == 0:
			places = re.split(r'[^\w]', location)
	except:
		pass
	return places


def get_country(country_map, place):

	for countryname in countrynames:

		if countryname['name']['common'].lower() == place.lower():
			country_map['is_country'].append(countryname['cca2'])
			return

		if countryname['cca2'].lower() == place.lower():
			country_map['is_country'].append(countryname['cca2'])
			return

		for k, translation in countryname['translations'].items():
			if translation['common'].lower() == place.lower():
				country_map['is_country'].append(countryname['cca2'])
				return

		for spelling in countryname['altSpellings']:
			if spelling.lower() == place.lower():
				country_map['is_country'].append(countryname['cca2'])
				return

	place_matches = geonames[geonames['name'].str.contains(place, na=False, case=False)]
	if not place_matches.empty:
		country_map['others'].append(place_matches.iloc[0]['country_code'])
		return


	place_matches = geonames[geonames['alternatenames'].str.contains(place.capitalize(), na=False)]
	if not place_matches.empty:
		country_map['others'].append(place_matches.iloc[0]['country_code'])
		return

	splitted_places = re.split(r'[^\w]', place)
	for splitted_place in splitted_places:
		if len(splitted_places) > 1:
			get_country(country_map, splitted_place)

def pick_country(country_map):
	if len(country_map['is_country'] + country_map['others']) == 0:
		return "CHECK"
	if len(country_map['is_country']) == 1:
		return country_map['is_country'][0]
	else:
		temp_list = country_map['is_country'] + country_map['others']
		return Counter(temp_list).most_common()[0][0]

if __name__ == "__main__":
	main()