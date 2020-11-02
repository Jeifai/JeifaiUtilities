import re
import json
import requests
import pandas as pd

locations = pd.read_csv('locations.csv')
geonames = pd.read_csv('geonames.txt', sep="	", header=None, low_memory=False)
geonames.columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
				'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code', 'admin2_code ', 
				'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification_date']
geonames.sort_values('population', ascending=False, inplace=True)
countrynames = requests.get("https://raw.githubusercontent.com/mledoze/countries/master/dist/countries.json").json()

def main():
	for i, location in locations.head(50).iterrows():
		print(location['location'])
		# clean_location = re.sub(r"[^\w,\w ]|_", ',', location['location'])
		# print(clean_location)
		places = get_places(location['location'])
		countries = {}
		for place in places:
			get_country(countries, place)
		print(json.dumps(countries, indent=4, sort_keys=True))
		# pick_country(countries)

def get_places(location):
	places = []
	for i in re.split(r'[^\w ]', location):
		temp_place = re.sub(r"[^\w,\w ]|_", ',', i.strip())
		if temp_place:
			places.append(temp_place)
	if len(places) == 0:
		places = re.split(r'[^\w]', location)
	return places


def get_country(countries, place):

	countries[place] = {}

	for countryname in countrynames:

		if countryname['name']['common'].lower() == place.lower():
			countries[place]['is_country'] = True
			countries[place]['country'] = countryname['cca2']
			countries[place]['population'] = None
			return

		if countryname['cca2'].lower() == place.lower():
			countries[place]['is_country'] = True
			countries[place]['country'] = countryname['cca2']
			countries[place]['population'] = None
			return

		for k, translation in countryname['translations'].items():
			if translation['common'].lower() == place.lower():
				countries[place]['is_country'] = True
				countries[place]['country'] = countryname['cca2']
				countries[place]['population'] = None
				return

		for spelling in countryname['altSpellings']:
			if spelling.lower() == place.lower():
				countries[place]['is_country'] = True
				countries[place]['country'] = countryname['cca2']
				countries[place]['population'] = None
				return

	place_matches = geonames[geonames['name'] == place]
	if not place_matches.empty:
		countries[place]['is_country'] = False
		countries[place]['country'] = place_matches.iloc[0]['country_code']
		countries[place]['population'] = int(place_matches.iloc[0]['population'])
		return


	place_matches = geonames[geonames['alternatenames'].str.contains(place, na=False)]
	if not place_matches.empty:
		countries[place]['is_country'] = False
		countries[place]['country'] = place_matches.iloc[0]['country_code']
		countries[place]['population'] = int(place_matches.iloc[0]['population'])

def pick_country(countries):
	for country in countries:
		if country['is_country']:
			return country['country']

if __name__ == "__main__":
	main()