from datetime import timedelta
import numpy as np
from typing import List, Union, Optional

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
        rx (float)   - x position in ECEF (m)
        ry (float)   - y position in ECEF (m)
        rz (float)   - z position in ECEF (m)
        vx (float)   - x velocity in ECEF (m/s)
        vy (float)   - y velocity in ECEF (m/s)
        vz (float)   - z velocity in ECEF (m/s)

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
               f'  x position:   {self.rx:.3f} (m)\n' \
               f'  y position:   {self.ry:.3f} (m)\n' \
               f'  z position:   {self.rz:.3f} (m)\n' \
               f'  x velocity:   {self.vx:.3f} (m/s)\n' \
               f'  y velocity:   {self.vy:.3f} (m/s)\n' \
               f'  z velocity:   {self.vz:.3f} (m/s)\n' \
               f')'

    def _calcMeasuredEciStateCovariance(self, assume_earth_rotation: bool=False) -> np.ndarray:
        '''
        Calculates the measured ECI state covariance.

        ### Inputs:
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (np.ndarray) measured ECI state covariance
        '''
        # measurement covariances are in ECEF, but need to be converted to ECI...
        # we just need to do a coordinate transform with: P_measured_eci = T × R × T^T
        T_ecef_to_eci = generateEcefToEciTransform(self.time, assume_earth_rotation)
        return T_ecef_to_eci @ self.R @ T_ecef_to_eci.T

    def toEciState(
            self,
            propagated_state:      Optional[EciState],
            assume_earth_rotation: bool=False
        ) -> EciState:
        '''
        Creates an ECI state from this GPS measurement and a propagated ECI state.

        ### Inputs:
        propagated_state (EciState | None) - propagated state to influence the
                                             generation of the ECI state
        assume_earth_rotation (bool)       - flag to turn "no Earth Rotation"
                                             assumption on and off

        ### Outputs:
        (EciState) ECI state from GPS measurement
        '''
        # generate the measured state
        measured_state = EcefState(
            self.time,
            self.rx,
            self.ry,
            self.rz,
            self.vx,
            self.vy,
            self.vz,
        ).toEciState(assume_earth_rotation)
        measured_state.P = self._calcMeasuredEciStateCovariance(assume_earth_rotation)
        # if there is no propagated state, trust the GPS alone
        if propagated_state is None:
            return measured_state
        # otherwise, apply Kalman update
        else:
            return kalmanUpdate(propagated_state, measured_state)

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
        ra (float)      - right ascension (rad)
        dec (float)     - declination (rad)
        ra_dot (float)  - right ascension rate (rad/s)
        dec_dot (float) - declination rate (rad/s)

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
        # trig shorthands
        c_ra  = np.cos(self.ra)
        s_ra  = np.sin(self.ra)
        c_dec = np.cos(self.dec)
        s_dec = np.sin(self.dec)
        # line-of-sight unit vector (spherical to cartesian)
        self.los = np.array([
            c_dec * c_ra,
            c_dec * s_ra,
            s_dec,
        ])
        # line-of-sight rate (los unit vector derivative wrt time)
        self.los_dot = np.array([
            (c_dec * -s_ra * self.ra_dot) + (-s_dec * c_ra * self.dec_dot),
            (c_dec *  c_ra * self.ra_dot) + (-s_dec * s_ra * self.dec_dot),
                                            ( c_dec        * self.dec_dot),
        ])
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

    def _calcMeasuredEciStateCovariance(
            self,
            r_veh_eci_mag:         float,
            assume_earth_rotation: bool=False,
        ) -> np.ndarray:
        '''
        Calculates the measured ECI state covariance.

        ### Inputs:
        r_veh_eci_mag (float)        - position magnitude of vehicle in ECI (m)
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (np.ndarray) measured ECI state covariance
        '''
        # trig shorthands
        c_ra  = np.cos(self.ra)
        s_ra  = np.sin(self.ra)
        c_dec = np.cos(self.dec)
        s_dec = np.sin(self.dec)
        # compute Jacobian of ECI state w.r.t. [ra, dec, ra_dot, dec_dot]...
        # we need Jacobians because measurement covariances are in angles,
        # but we need them to be in position/velocity
        J = np.zeros((6, 4))  # 6 state variables, 4 measurement variables
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
        J[3, 0] = r_veh_eci_mag * (-self.ra_dot * c_dec * c_ra +  self.dec_dot * s_dec * s_ra)
        J[4, 0] = r_veh_eci_mag * (-self.ra_dot * c_dec * s_ra + -self.dec_dot * s_dec * c_ra)
        J[5, 0] = 0
        # dVel/dDec
        J[3, 1] = r_veh_eci_mag * ( self.ra_dot * s_dec * s_ra + -self.dec_dot * c_dec * c_ra)
        J[4, 1] = r_veh_eci_mag * (-self.ra_dot * s_dec * c_ra + -self.dec_dot * c_dec * s_ra)
        J[5, 1] = r_veh_eci_mag * (                              -self.dec_dot * s_dec       )
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
        # measured ECI state covariance: P_measured_eci = J * R * J^T
        return J @ self.R @ J.T

    def toEciState(
            self,
            propagated_state:      EciState,
            assume_earth_rotation: bool=False,
        ) -> EciState:
        '''
        Creates an ECI state from this ground station measurement and a propagated ECI state.

        Ground station measurements do not have enough information on their own
        to generate an ECI state, so the function REQUIRES a propagated state as an input.

        ### Inputs:
        propagated_state (EciState | None) - propagated state to influence the
                                             generation of the ECI state
        assume_earth_rotation (bool)       - flag to turn "no Earth Rotation"
                                             assumption on and off

        ### Outputs:
        (EciState) ECI state from ground station measurement
        '''
        # get propogated position magnitude of vehicle in ECI for state estimation
        r_veh_eci_mag = np.linalg.norm(propagated_state.state[0:3], axis=0)
        # ECI position
        r_veh_eci = r_veh_eci_mag * self.los
        # angular velocity
        v_angular = r_veh_eci_mag * self.los_dot
        # ECI velocity
        if assume_earth_rotation:
            # Earth's angular velocity vector in ECI
            omega_earth = np.array([0, 0, Constants.W_EARTH])
            # add velocity due to Earth's rotation
            v_veh_eci = v_angular + np.cross(omega_earth, r_veh_eci)
        else:
            # for non-rotating Earth, the velocity is the angular rate
            v_veh_eci = v_angular
        # generate the measured state
        measured_state = EciState(
            self.time,
            *r_veh_eci,
            *v_veh_eci,
            P=self._calcMeasuredEciStateCovariance(r_veh_eci_mag, assume_earth_rotation),
        )
        # apply Kalman update
        return kalmanUpdate(propagated_state, measured_state)

def measurementsToEciStates(
        measurements:          List[Union[GpsMeasurement, GroundStationMeasurement]],
        assume_earth_rotation: bool=False,
    ) -> List[EciState]:
    '''
    Generate a list of ECI states from a list of measurements.

    ### Inputs:
    measurements (List[GpsMeasurement | GroundStationMeasurement]) - GPS and ground station measurements
    assume_earth_rotation (bool)                                   - flag to turn "no Earth Rotation"
                                                                     assumption on and off

    ### Outputs:
    (List[EciState]) ECI state for each measurement
    '''
    states = []
    propagated_state = None
    for measurement in sorted(measurements, key=lambda m: m.time):
        # propagate most recent state to measurement time (if we have a state)
        if len(states) > 0:
            curr_state = states[-1]
            dt = (measurement.time - curr_state.time).total_seconds()
            propagated_state = propagateState(curr_state, dt)
        # GPS measurements
        if isinstance(measurement, GpsMeasurement):
            states.append(measurement.toEciState(propagated_state, assume_earth_rotation))
        # ground station measurements
        elif isinstance(measurement, GroundStationMeasurement):
            # we need the radius from a propogated state to generate a new ECI state
            if propagated_state is None:
                continue
            states.append(measurement.toEciState(propagated_state, assume_earth_rotation))
        else:
            raise TypeError(f'Invalid measurement type {type(measurement)}')
    return states

def propagateState(state: EciState, dt: float) -> EciState:
    '''
    Propagate state and covariance forward in time using orbital dynamics.

    ### Inputs:
    state (EciState) - current state
    dt (float)       - time step in seconds

    ### Outputs:
    (Tuple[EciState, np.ndarray]) Propagated state and covariance
    '''
    # TODO: use Kepler's problem to propagate orbit
    new_state = state.state
    # TODO: state transition matrix (linearization of dynamics)
    Phi = np.eye(6) # computeStateTransitionMatrix(state.state, dt)
    # process noise (model uncertainty)... I'm feeling super certain :P
    Q = np.zeros((6, 6))
    # propagate covariance
    P_new = Phi @ state.P @ Phi.T + Q
    # create new state
    return EciState(state.time + timedelta(seconds=dt), *new_state, P=P_new)

def kalmanUpdate(
        prior_state:    EciState,
        measured_state: EciState,
    ) -> EciState:
    '''
    Perform Kalman update using measurement.

    ### Inputs:
    prior_state (EciState)    - propagated state
    measured_state (EciState) - measured state

    ### Outputs:
    (EciState) updated state
    '''
    # measurement Jacobian... for direct state measurements like prior_state
    # and measured_state (both are ECI pos/vel), H is identity
    H = np.eye(6)
    # innovation covariance
    S = H @ prior_state.P @ H.T + measured_state.P
    # Kalman gain
    K = prior_state.P @ H.T @ np.linalg.inv(S)
    # state update
    updated_state = prior_state.state + K @ (measured_state.state - H @ prior_state.state)
    # covariance update using Joseph form for numerical stability
    I = np.eye(6)
    P_updated = (I - K @ H) @ prior_state.P @ (I - K @ H).T + K @ measured_state.P @ K.T
    # create updated state
    return EciState(measured_state.time, *updated_state, P=P_updated)
