from datetime import datetime, timedelta
import numpy as np
from typing import List, Union, Tuple

from .consts import Constants
from .ecef import EcefState, generateEcefToEciTransform
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
        ).toEciState(assume_earth_rotation)

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
        elif self.id == Constants.ID_GS2:
            self.R = Constants.R_GS2
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
            r_veh_eci_mag:             float,
            assume_earth_rotation: bool=False,
        ) -> EciState:
        '''
        Creates an equivalent ECI state from this ground station measurement.

        Ground station measurements do not have enough information on their own
        to generate an ECI state, so the function requires r_veh_eci_mag as an input.

        ### Inputs:
        r_veh_eci_mag (float)        - position magnitude of vehicle in ECI (km)
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
        # line-of-sight (spherical to cartesian)
        los = np.array([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])
        # line-of-sight rate (derivative wrt time)
        los_dot = np.array([
            (np.cos(dec_rad) * -np.sin(ra_rad) * ra_dot_rad) + (-np.sin(dec_rad) * np.cos(ra_rad) * dec_dot_rad),
            (np.cos(dec_rad) *  np.cos(ra_rad) * ra_dot_rad) + (-np.sin(dec_rad) * np.sin(ra_rad) * dec_dot_rad),
                                                               ( np.cos(dec_rad)                  * dec_dot_rad),
        ])
        # ECI position
        r_veh_eci = r_veh_eci_mag * los
        # angular velocity
        v_angular = r_veh_eci_mag * los_dot
        # ECI velocity
        if assume_earth_rotation:
            # Earth's angular velocity vector in ECI
            omega_earth = np.array([0, 0, Constants.W_EARTH])
            # add velocity due to Earth's rotation
            v_veh_eci = v_angular + np.cross(omega_earth, r_veh_eci)
        else:
            # for non-rotating Earth, the velocity is the angular rate
            v_veh_eci = v_angular
        # return the state
        return EciState(*r_veh_eci, *v_veh_eci)

def measurementsToEciStates(
        measurements:          List[Union[GpsMeasurement, GroundStationMeasurement]],
        assume_earth_rotation: bool=False,
    ) -> List[Tuple[EciState, np.ndarray, datetime]]:
    '''
    Generate a list of ECI states from a list of measurements.

    ### Inputs:
    measurements (List[GpsMeasurement | GroundStationMeasurement]) - GPS and ground station measurements
                                                                     which must be sorted by time
    assume_earth_rotation (bool)                                   - flag to turn "no Earth Rotation"
                                                                     assumption on and off

    ### Outputs:
    (List[Tuple[EciState, np.ndarray, datetime]]) ECI state, covariance, and time for each measurement
    '''
    # create variables to keep track of
    states = []
    # use previous r_veh_eci_mag for ground station ECI state calculation...
    # this currently assumes that |r| isn't changing...
    # TODO: make this better - what if it's not a circular orbit?
    r_veh_eci_mag = None
    # loop over each measurement and get the ECI state
    for measurement in measurements:
        # GPS measurements
        if isinstance(measurement, GpsMeasurement):
            # convert GPS measurement to ECI state
            eci_state = measurement.toEciState(assume_earth_rotation=assume_earth_rotation)
            # get position magnitude for ground station measurements
            r_veh_eci_mag = np.linalg.norm([eci_state.rx, eci_state.ry, eci_state.rz], axis=0)
            # covariance measurements are in ECEF, but need to be converted to ECI...
            # we just need to do a coordinate transform with: P_eci = T × P_ecef × T^T
            T_ecef_to_eci = generateEcefToEciTransform(measurement.time, assume_earth_rotation)
            P_eci = T_ecef_to_eci @ measurement.R @ T_ecef_to_eci.T
            states.append((eci_state, P_eci, measurement.time))
        # ground station measurements
        elif isinstance(measurement, GroundStationMeasurement) and r_veh_eci_mag is not None:
            # convert angles to radians for calculation
            ra_rad = np.deg2rad(measurement.ra)
            dec_rad = np.deg2rad(measurement.dec)
            ra_dot_rad = np.deg2rad(measurement.ra_dot)
            dec_dot_rad = np.deg2rad(measurement.dec_dot)
            # compute ECI state
            eci_state = measurement.toEciState(r_veh_eci_mag, assume_earth_rotation)
            # compute Jacobian of ECI state w.r.t. [ra, dec, ra_dot, dec_dot]...
            # we need Jacobians because covariance measurements are in angles,
            # but we need them to be in position/velocity
            J = np.zeros((6, 4))  # 6 state variables, 4 measurement variables
            c_dec = np.cos(dec_rad)
            s_dec = np.sin(dec_rad)
            c_ra  = np.cos(ra_rad)
            s_ra  = np.sin(ra_rad)
            # position partials (r = r_mag * [cos(dec)cos(ra), cos(dec)sin(ra), sin(dec)])
            # dPos/dRA
            J[0, 0] = r_veh_eci_mag * c_dec * -s_ra
            J[1, 0] = r_veh_eci_mag * c_dec *  c_ra
            J[2, 0] = 0
            # dPos/dDec
            J[0, 1] = r_veh_eci_mag * -s_dec * c_ra
            J[1, 1] = r_veh_eci_mag * -s_dec * s_ra
            J[2, 1] = r_veh_eci_mag *  c_dec
            # dPos/dRa_dot
            J[0, 2] = 0
            J[1, 2] = 0
            J[2, 2] = 0
            # dPos/dDec_dot
            J[0, 3] = 0
            J[1, 3] = 0
            J[2, 3] = 0
            # velocity partials due to angular velocity (v = r_mag * [
            #   -ra_dot * cos(dec)sin(ra) + -dec_dot * sin(dec)cos(ra),
            #    ra_dot * cos(dec)cos(ra) + -dec_dot * sin(dec)sin(ra),
            #                                dec_dot * cos(dec)       ,
            #])
            # dVel/dRA
            J[3, 0] = r_veh_eci_mag * (-ra_dot_rad * c_dec * c_ra +  dec_dot_rad * s_dec * s_ra)
            J[4, 0] = r_veh_eci_mag * (-ra_dot_rad * c_dec * s_ra + -dec_dot_rad * s_dec * c_ra)
            J[5, 0] = 0
            # dVel/dDec
            J[3, 1] = r_veh_eci_mag * ( ra_dot_rad * s_dec * s_ra + -dec_dot_rad * c_dec * c_ra)
            J[4, 1] = r_veh_eci_mag * (-ra_dot_rad * s_dec * c_ra + -dec_dot_rad * c_dec * s_ra)
            J[5, 1] = r_veh_eci_mag * (                             -dec_dot_rad * s_dec       )
            # dVel/dRA_dot
            J[3, 2] = r_veh_eci_mag * (-c_dec * s_ra)
            J[4, 2] = r_veh_eci_mag * ( c_dec * c_ra)
            J[5, 2] = 0
            # dVel/dDec_dot
            J[3, 3] = r_veh_eci_mag * (-s_dec * c_ra)
            J[4, 3] = r_veh_eci_mag * (-s_dec * s_ra)
            J[5, 3] = r_veh_eci_mag * ( c_dec       )
            # velocity partials due to Earth's rotation (v = W_EARTH * r_mag * [
            #   -cos(dec)sin(ra),
            #    cos(dec)cos(ra),
            #                  0,
            # ])
            if assume_earth_rotation:
                # dVel/dRA
                J[3, 0] += Constants.W_EARTH * r_veh_eci_mag * -c_dec *  c_ra
                J[4, 0] += Constants.W_EARTH * r_veh_eci_mag *  c_dec * -s_ra
                J[5, 0] += 0
                # dVel/dDec
                J[3, 1] += Constants.W_EARTH * r_veh_eci_mag *  s_dec *  s_ra
                J[4, 1] += Constants.W_EARTH * r_veh_eci_mag * -s_dec *  c_ra
                J[5, 1] += 0
                # dVel/dRA_dot
                J[3, 2] += 0
                J[4, 2] += 0
                J[5, 2] += 0
                # dVel/dDec_dot
                J[3, 3] += 0
                J[4, 3] += 0
                J[5, 3] += 0
            # transform covariance: P_eci = J * R * J^T
            P_eci = J @ measurement.R @ J.T
            states.append((eci_state, P_eci, measurement.time))
        else:
            raise TypeError('This function only support GpsMeasurement and GroundStationMeasurement types')
    return states
