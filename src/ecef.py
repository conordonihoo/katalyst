import numpy as np
import spiceypy as spice
from datetime import datetime
from typing import Tuple

from .consts import Constants
from .eci import EciState, loadKernels

class EcefState:

    def __init__(
            self,
            time: datetime,
            rx:   float,
            ry:   float,
            rz:   float,
            vx:   float,
            vy:   float,
            vz:   float,
        ) -> None:
        '''
        The EcefState object initializer.

        ### Inputs:
        time (datetime) - time of state (--)
        rx (float)      - x position in ECEF (km)
        ry (float)      - y position in ECEF (km)
        rz (float)      - z position in ECEF (km)
        vx (float)      - x velocity in ECEF (km/s)
        vy (float)      - y velocity in ECEF (km/s)
        vz (float)      - z velocity in ECEF (km/s)

        ### Outputs:
        None
        '''
        self.time = time
        self.rx   = rx
        self.ry   = ry
        self.rz   = rz
        self.vx   = vx
        self.vy   = vy
        self.vz   = vz
        return

    def __str__(self) -> str:
        '''
        Return this object in string format.

        ### Inputs:
        None

        ### Outputs:
        (str) representing this object
        '''
        return f'EcefState(\n' \
               f'  time:         {self.time}\n' \
               f'  x position:   {self.rx:.3f} (km)\n' \
               f'  y position:   {self.ry:.3f} (km)\n' \
               f'  z position:   {self.rz:.3f} (km)\n' \
               f'  x velocity:   {self.vx:.3f} (km/s)\n' \
               f'  y velocity:   {self.vy:.3f} (km/s)\n' \
               f'  z velocity:   {self.vz:.3f} (km/s)\n' \
               f')'

    def toEciState(self, assume_earth_rotation: bool=False) -> EciState:
        '''
        Creates an equivalent ECI state from this ECEF state.

        ### Inputs:
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (EciState) equivalent to this ECEF state
        '''
        # put state into np array
        ecef_state = np.array([self.rx, self.ry, self.rz, self.vx, self.vy, self.vz])
        # use kernels to do ECEF to ECI math (too lazy to poorly reinvent the wheel)
        if assume_earth_rotation:
            # use spice to get the ECEF to ECI transformation matrix
            loadKernels()
            days_since_J2000_in_UTC = (self.time - Constants.J2000_EPOCH).total_seconds() / 86400
            secs_since_J2000_in_TDB = spice.str2et('jd ' + str(Constants.J2000_JD + days_since_J2000_in_UTC))
            T_ecef_to_eci = spice.sxform('ITRF93', 'J2000', secs_since_J2000_in_TDB)
            eci_state = T_ecef_to_eci @ ecef_state
        # assuming no Earth rotation, ECI state == ECEF state
        else:
            eci_state = ecef_state
        return EciState(*eci_state)

def geodeticToEcef(
        lat: float,
        lon: float,
        alt: float,
    ) -> Tuple[float, float, float]:
    '''
    Convert geodetic coordinates (latitude, longitude, altitude)
    to ECEF coordinates.

    ### Inputs:
    lat (float) - geodetic latitude (deg)
    lon (float) - longitude (deg)
    alt (float) - altitude (km)

    ### Outputs:
    (Tuple[float, float, float]) x, y, z coordinates in ECEF (km)
    '''
    # convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    # for a spherical Earth, the conversion to the point's location in
    # spherical coordinates is just Earth's radius plus altitude
    radius = Constants.R_EARTH + alt
    # calculate ECEF coordinates using spherical coordinate conversion
    # source: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z
