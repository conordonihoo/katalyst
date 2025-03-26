import numpy as np
import spiceypy as spice
from datetime import datetime
from typing import Tuple

from .utils import Constants
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
        rx (float)      - x position in ECEF (m)
        ry (float)      - y position in ECEF (m)
        rz (float)      - z position in ECEF (m)
        vx (float)      - x velocity in ECEF (m/s)
        vy (float)      - y velocity in ECEF (m/s)
        vz (float)      - z velocity in ECEF (m/s)

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
               f'  x position:   {self.rx / 1000:.3f} (km)\n' \
               f'  y position:   {self.ry / 1000:.3f} (km)\n' \
               f'  z position:   {self.rz / 1000:.3f} (km)\n' \
               f'  x velocity:   {self.vx / 1000:.3f} (km/s)\n' \
               f'  y velocity:   {self.vy / 1000:.3f} (km/s)\n' \
               f'  z velocity:   {self.vz / 1000:.3f} (km/s)\n' \
               f')'
    @property
    def state(self) -> np.ndarray:
        '''
        Read-only alias for the state variables.

        ### Inputs:
        None

        ### Outputs:
        (np.ndarray) containing state variables
        '''
        return np.array([self.rx, self.ry, self.rz, self.vx, self.vy, self.vz])


    def toEciState(self, assume_earth_rotation: bool=False) -> EciState:
        '''
        Creates an equivalent ECI state from this ECEF state.

        ### Inputs:
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (EciState) equivalent to this ECEF state
        '''
        T_ecef_to_eci = generateEcefToEciTransform(self.time, assume_earth_rotation)
        eci_state = T_ecef_to_eci @ self.state
        return EciState(self.time, *eci_state)

def generateEcefToEciTransform(time: datetime, assume_earth_rotation: bool=False) -> np.ndarray:
    '''
    Generates the ECEF to ECI transformation matrix at a given time.

    ### Inputs:
    time (datetime)              - time at which the transformation occurs
    assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                    assumption on and off

    ### Outputs:
    (np.ndarray) of the ECEF to ECI transformation matrix
    '''
    # assuming no Earth rotation, ECEF == ECI
    T_ecef_to_eci = np.eye(6)
    # use spice to get the actual ECEF to ECI transformation matrix
    if assume_earth_rotation:
        loadKernels()
        days_since_J2000_in_UTC = (time - Constants.J2000_EPOCH).total_seconds() / 86400
        secs_since_J2000_in_TDB = spice.str2et('jd ' + str(Constants.J2000_JD + days_since_J2000_in_UTC))
        T_ecef_to_eci = spice.sxform('ITRF93', 'J2000', secs_since_J2000_in_TDB)
    return T_ecef_to_eci

def geodeticToEcef(
        lat: float,
        lon: float,
        alt: float,
    ) -> Tuple[float, float, float]:
    '''
    Convert geodetic coordinates (latitude, longitude, altitude)
    to ECEF coordinates.

    ### Inputs:
    lat (float) - geodetic latitude (rad)
    lon (float) - longitude (rad)
    alt (float) - altitude (m)

    ### Outputs:
    (Tuple[float, float, float]) x, y, z coordinates in ECEF (m)
    '''
    # for a spherical Earth, the conversion to the point's location in
    # spherical coordinates is just Earth's radius plus altitude
    radius = Constants.R_EARTH + alt
    # calculate ECEF coordinates using spherical coordinate conversion
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z
