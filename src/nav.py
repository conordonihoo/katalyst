from datetime import datetime, timedelta
import numpy as np
from typing import List, Union, Tuple

from .consts import Constants
from .ecef import EcefState
from .eci import EciState

class GpsMeasurement:

    def __init__(
            self,
            time: float,
            id:   str,
            rx:   float,
            ry:   float,
            rz:   float,
            vx:   float,
            vy:   float,
            vz:   float,
        ) -> None:
        '''
        The GpsMeasurement object initializer.

        ### Inputs:
        time (float) - time after T0 epoch (s)
        id (str)     - GPS ID (--)
        rx (float)   - x position in ECEF (km)
        ry (float)   - y position in ECEF (km)
        rz (float)   - z position in ECEF (km)
        vx (float)   - x velocity in ECEF (km/s)
        vy (float)   - y velocity in ECEF (km/s)
        vz (float)   - z velocity in ECEF (km/s)

        ### Outputs:
        None
        '''
        self.time = Constants.T0_EPOCH + timedelta(seconds=time)
        self.id   = id
        self.rx   = rx
        self.ry   = ry
        self.rz   = rz
        self.vx   = vx
        self.vy   = vy
        self.vz   = vz
        # set the covariance
        self.R    = Constants.R_GPS
        return

    def __str__(self) -> str:
        '''
        Return this object in string format.

        ### Inputs:
        None

        ### Outputs:
        (str) representing this object
        '''
        return f'GpsMeasurement(\n' \
               f'  time:         {self.time}\n' \
               f'  id:           {self.id}\n' \
               f'  x position:   {self.rx:.3f} (km)\n' \
               f'  y position:   {self.ry:.3f} (km)\n' \
               f'  z position:   {self.rz:.3f} (km)\n' \
               f'  x velocity:   {self.vx:.3f} (km/s)\n' \
               f'  y velocity:   {self.vy:.3f} (km/s)\n' \
               f'  z velocity:   {self.vz:.3f} (km/s)\n' \
               f')'

    def toEciState(self, assume_earth_rotation: bool=False) -> EciState:
        '''
        Creates an equivalent ECI state from this GPS measurement.

        ### Inputs:
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (EciState) ECI state from GPS measurement
        '''
        return EcefState(
            self.time,
            self.rx,
            self.ry,
            self.rz,
            self.vx,
            self.vy,
            self.vz
        ).toEciState(assume_earth_rotation=assume_earth_rotation)

class GroundStationMeasurement:

    def __init__(
            self,
            time:    float,
            id:      str,
            ra:      float,
            dec:     float,
            ra_dot:  float,
            dec_dot: float,
        ) -> None:
        '''
        The GroundStationMeasurement object initializer.

        ### Inputs:
        time (float)    - time after T0 epoch (s)
        id (str)        - ground station ID (--)
        ra (float)      - right ascension (deg)
        dec (float)     - declination (deg)
        ra_dot (float)  - right ascension rate (deg/s)
        dec_dot (float) - declination rate (deg/s)

        ### Outputs:
        None
        '''
        self.time    = Constants.T0_EPOCH + timedelta(seconds=time)
        self.id      = id
        self.ra      = ra
        self.dec     = dec
        self.ra_dot  = ra_dot
        self.dec_dot = dec_dot
        # set the covariance and ground station location
        if self.id == Constants.ID_GS1:
            self.R = Constants.R_GS1
            self.LLA = Constants.LLA_GS1
        elif self.id == Constants.ID_GS2:
            self.R = Constants.R_GS2
            self.LLA = Constants.LLA_GS2
        else:
            raise ValueError(f'Unknown ground station ID: {self.id}')
        return

    def __str__(self) -> str:
        '''
        Return this object in string format.

        ### Inputs:
        None

        ### Outputs:
        (str) representing this object
        '''
        return f'GroundStationMeasurement(\n' \
               f'  time:  {self.time}\n' \
               f'  id:    {self.id}\n' \
               f'  RA:    {self.ra:.3f} (deg)\n' \
               f'  Dec:   {self.dec:.3f} (deg)\n' \
               f'  dRA:   {self.ra_dot:.3f} (deg/s)\n' \
               f'  dDec:  {self.dec_dot:.3f} (deg/s)\n' \
               f')'

    def toEciState(
            self,
            r_mag_eci:             float,
            assume_earth_rotation: bool=False,
        ) -> EciState:
        '''
        Creates an equivalent ECI state from this ground station measurement.

        Ground station measurements do not have enough information on their own
        to generate an ECI state, so the function requires r_mag_eci as an input.

        ### Inputs:
        r_mag_eci (float)            - position magnitude of vehicle in ECI (km)
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (EciState) ECI state from ground station measurement
        '''
        # convert angles from degrees to radians
        ra_rad = np.deg2rad(self.ra)
        dec_rad = np.deg2rad(self.dec)
        ra_dot_rad = np.deg2rad(self.ra_dot)
        dec_dot_rad = np.deg2rad(self.dec_dot)
        # line-of-sight (spherical to cartesian) and line-of-sight rate (partial derivatives)
        los = np.array([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])
        los_dot = np.array([
            -np.sin(ra_rad) * np.cos(dec_rad) * ra_dot_rad - np.cos(ra_rad) * np.sin(dec_rad) * dec_dot_rad,
            np.cos(ra_rad) * np.cos(dec_rad) * ra_dot_rad - np.sin(ra_rad) * np.sin(dec_rad) * dec_dot_rad,
            np.cos(dec_rad) * dec_dot_rad,
        ])
        # ECI position
        r_eci = r_mag_eci * los
        # angular velocity
        v_angular = r_mag_eci * los_dot
        # ECI velocity
        if assume_earth_rotation:
            # Earth's angular velocity vector in ECI
            omega_earth = np.array([0, 0, Constants.W_EARTH])
            # add velocity due to Earth's rotation
            v_eci = v_angular + np.cross(omega_earth, r_eci)
        else:
            # for non-rotating Earth, the velocity is the angular rate
            v_eci = v_angular
        # return the state
        return EciState(*r_eci, *v_eci)

def measurementsToEciStates(
        measurements:          List[Union[GpsMeasurement, GroundStationMeasurement]],
        assume_earth_rotation: bool=False,
    ) -> List[Tuple[EciState, np.ndarray, datetime]]:
    '''
    Generate a list of ECI states from a list of measurements.

    ### Inputs:
    measurements (List[GpsMeasurement | GroundStationMeasurement]) - GPS and ground station measurements
    assume_earth_rotation (bool)                                   - flag to turn "no Earth Rotation"
                                                                     assumption on and off

    ### Outputs:
    (List[Tuple[EciState, np.ndarray, datetime]]) ECI state, covariance, and time for each measurement
    '''
    return []
