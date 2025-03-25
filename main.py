import argparse
from typing import List, Union

from src.consts import Constants
from src.ecef import *
from src.eci import *
from src.nav import *

def parseArgs() -> argparse.Namespace:
    desc = 'main.py script to generate a navigation estimation at some time'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rotating-earth', action='store_true',
                        help='Flag to simulate a rotating Earth')
    args = parser.parse_args()
    return args

def parseCsv(csv_path: str) -> List[Union[GpsMeasurement, GroundStationMeasurement]]:
    '''
    Parse GPS and ground station measurements from a CSV file.

    ### Inputs:
    csv_path (str) - Path to the CSV file

    ### Outputs:
    (List[GpsMeasurement | GroundStationMeasurement]) in the CSV
    '''
    import csv
    measurements = []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            time = float(row['time'])
            sensor_id = row['sensor_id']
            if sensor_id == Constants.ID_GPS:
                # Parse GPS measurement
                rx = float(row['r_x_km'])
                ry = float(row['r_y_km'])
                rz = float(row['r_z_km'])
                vx = float(row['v_x_km/s'])
                vy = float(row['v_y_km/s'])
                vz = float(row['v_z_km/s'])
                measurements.append(GpsMeasurement(
                    time, sensor_id, rx, ry, rz, vx, vy, vz
                ))
            elif sensor_id in [Constants.ID_GS1, Constants.ID_GS2]:
                # Parse ground station measurement
                ra = float(row['ra'])
                dec = float(row['dec'])
                ra_dot = float(row['ra_rate'])
                dec_dot = float(row['dec_rate'])
                measurements.append(GroundStationMeasurement(
                    time, sensor_id, ra, dec, ra_dot, dec_dot
                ))
            else:
                raise ValueError(f'Unrecognized sensor ID "{sensor_id}"')
    return measurements

def main(assume_rotating_earth: bool) -> None:
    '''
    Main function.

    ### Inputs:
    assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                    assumption on and off

    ### Outputs:
    None
    '''

    # Get Measurements
    measurements = parseCsv('./data/measurements.csv')
    return

if __name__ == '__main__':
    args = parseArgs()
    main(assume_rotating_earth=args.rotating_earth)
