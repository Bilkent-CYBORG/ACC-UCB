import pickle

import dateutil
import pandas as pd

"""
This script will parse the Gowalla location checkin dataset and extract the data and nomralize the latitude/longitude
"""

saved_df_filename = 'gowalla_df'  # the name of the file where the dataframe will be saved
file_name = 'Gowalla_totalCheckins.txt'  # the name of the gowalla dataset file
load_from_pickle = False  # this is used for debugging, leave it as False


def parse_line(line: str):
    line_arr = line.split("\t")
    user = int(line_arr[0])
    time = dateutil.parser.parse(line_arr[1])
    lat = float(line_arr[2])
    long = float(line_arr[3])
    location = int(line_arr[4])

    return user, time, lat, long, location


if __name__ == '__main__':
    colnames = ['user', 'checkin', 'latitude', 'longitude', 'location']

    if load_from_pickle:
        with open(saved_df_filename, 'rb') as input_file:
            df = pickle.load(input_file)
    else:
        with open(file_name) as f:
            df = pd.DataFrame([parse_line(l) for l in f], columns=colnames)

        # Normalize lat
        min_lat = df['latitude'].min()
        max_lat = df['latitude'].max()
        df['latitude'] = (df['latitude'] - min_lat) / (max_lat - min_lat)

        # Normalize long
        min_long = df['longitude'].min()
        max_long = df['longitude'].max()
        df['longitude'] = (df['longitude'] - min_long) / (max_long - min_long)

        with open(saved_df_filename, 'wb') as output:
            pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)

    print('Done loading Gowalla dataset')
