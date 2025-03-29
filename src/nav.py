from datetime import timedelta
import numpy as np
from typing import List, Union, Optional

from .utils import modPos, Constants
from .ecef import EcefState, generateEcefToEciTransform
from .eci import EciState, KeplerianState

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
            r_prop_mag:            float,
            v_prop:                np.ndarray,
            assume_earth_rotation: bool=False,
        ) -> np.ndarray:
        '''
        Calculates the measured ECI state covariance.

        ### Inputs:
        r_prop_mag (float)           - position magnitude of vehicle in ECI (m)
        v_prop (np.ndarray)          - velocity of vehicle in ECI (m/s)
        assume_earth_rotation (bool) - flag to turn "no Earth Rotation"
                                       assumption on and off

        ### Outputs:
        (np.ndarray) measured ECI state covariance
        '''
        # trig shorthands
        c_ra   = np.cos(self.ra)
        s_ra   = np.sin(self.ra)
        c_dec  = np.cos(self.dec)
        s_dec  = np.sin(self.dec)
        c_2ra  = np.cos(2 * self.ra)
        s_2ra  = np.sin(2 * self.ra)
        c_2dec = np.cos(2 * self.dec)
        s_2dec = np.sin(2 * self.dec)
        # compute Jacobian of ECI state w.r.t. [ra, dec, ra_dot, dec_dot]...
        # we need Jacobians because measurement covariances are in angles,
        # but we need them to be in position/velocity
        J = np.zeros((6, 4))  # 6 state variables, 4 measurement variables
        # position partials (r = r_prop_mag * [cos(dec)cos(ra), cos(dec)sin(ra), sin(dec)])
        # dPos/dRA
        J[0, 0] = r_prop_mag * c_dec * -s_ra
        J[1, 0] = r_prop_mag * c_dec *  c_ra
        J[2, 0] = 0
        # dPos/dDec
        J[0, 1] = r_prop_mag * -s_dec * c_ra
        J[1, 1] = r_prop_mag * -s_dec * s_ra
        J[2, 1] = r_prop_mag *  c_dec
        # dPos/dRa_dot
        J[0, 2] = 0
        J[1, 2] = 0
        J[2, 2] = 0
        # dPos/dDec_dot
        J[0, 3] = 0
        J[1, 3] = 0
        J[2, 3] = 0
        # velocity partials due to tangential velocity (v = [
        #   v_px * cos(dec)cos(ra)cos(dec)cos(ra) + v_py * cos(dec)sin(ra)cos(dec)cos(ra) + v_pz * sin(dec)cos(dec)cos(ra) + r_prop_mag * -ra_dot * cos(dec)sin(ra) + r_prop_mag * -dec_dot * sin(dec)cos(ra),
        #   v_px * cos(dec)cos(ra)cos(dec)sin(ra) + v_py * cos(dec)sin(ra)cos(dec)sin(ra) + v_pz * sin(dec)cos(dec)sin(ra) + r_prop_mag *  ra_dot * cos(dec)cos(ra) + r_prop_mag * -dec_dot * sin(dec)sin(ra),
        #   v_px * cos(dec)cos(ra)sin(dec)        + v_py * cos(dec)sin(ra)sin(dec)        + v_pz * sin(dec)sin(dec)        +                                          r_prop_mag *  dec_dot * cos(dec)       ,
        #])
        # dVel/dRA
        v_px, v_py, v_pz = v_prop
        J[3, 0] = v_px * c_dec * c_dec * -s_2ra + v_py * c_dec * c_dec * c_2ra + v_pz * s_dec * c_dec * -s_ra + r_prop_mag * (-self.ra_dot * c_dec * c_ra +  self.dec_dot * s_dec * s_ra)
        J[4, 0] = v_px * c_dec * c_dec *  c_2ra + v_py * c_dec * c_dec * s_2ra + v_pz * s_dec * c_dec *  c_ra + r_prop_mag * (-self.ra_dot * c_dec * s_ra + -self.dec_dot * s_dec * c_ra)
        J[5, 0] = v_px * c_dec * -s_ra *  s_dec + v_py * c_dec *  c_ra * s_dec
        # dVel/dDec
        J[3, 1] = v_px * c_ra * -s_2dec * c_ra + v_py * s_ra * -s_2dec * c_ra + v_pz * c_2dec * c_ra + r_prop_mag * ( self.ra_dot * s_dec * s_ra + -self.dec_dot * c_dec * c_ra)
        J[4, 1] = v_px * c_ra * -s_2dec * s_ra + v_py * s_ra * -s_2dec * s_ra + v_pz * c_2dec * s_ra + r_prop_mag * (-self.ra_dot * s_dec * c_ra + -self.dec_dot * c_dec * s_ra)
        J[5, 1] = v_px * c_ra *  c_2dec        + v_py * s_ra *  c_2dec        + v_pz * s_2dec        + r_prop_mag * (                              -self.dec_dot * s_dec       )
        # dVel/dRA_dot
        J[3, 2] = r_prop_mag * (-c_dec * s_ra)
        J[4, 2] = r_prop_mag * ( c_dec * c_ra)
        J[5, 2] = 0
        # dVel/dDec_dot
        J[3, 3] = r_prop_mag * (-s_dec * c_ra)
        J[4, 3] = r_prop_mag * (-s_dec * s_ra)
        J[5, 3] = r_prop_mag * ( c_dec       )
        # velocity partials due to Earth's rotation (v = W_EARTH * r_prop_mag * [
        #   -cos(dec)sin(ra),
        #    cos(dec)cos(ra),
        #                  0,
        # ])
        if assume_earth_rotation:
            # dVel/dRA
            J[3, 0] += Constants.W_EARTH * r_prop_mag * -c_dec *  c_ra
            J[4, 0] += Constants.W_EARTH * r_prop_mag *  c_dec * -s_ra
            J[5, 0] += 0
            # dVel/dDec
            J[3, 1] += Constants.W_EARTH * r_prop_mag *  s_dec *  s_ra
            J[4, 1] += Constants.W_EARTH * r_prop_mag * -s_dec *  c_ra
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
        # get propagated position and velocity to inform measurements
        # to calculate the radial distance and velocity
        r_prop = propagated_state.state[0:3]
        v_prop = propagated_state.state[3:6]
        r_prop_mag = np.linalg.norm(r_prop, axis=0)
        # trig shorthands
        c_ra  = np.cos(self.ra)
        s_ra  = np.sin(self.ra)
        c_dec = np.cos(self.dec)
        s_dec = np.sin(self.dec)
        # position unit vector (spherical to cartesian)
        r_uv = np.array([
            c_dec * c_ra,
            c_dec * s_ra,
            s_dec,
        ])
        # ECI position
        r_veh_eci = r_prop_mag * r_uv
        # ECI velocity (derivative of position wrt time)
        dr_prop_mag = np.dot(v_prop, r_uv)
        v_veh_eci = np.array([
            (dr_prop_mag * c_dec * c_ra) + (r_prop_mag * c_dec * -s_ra * self.ra_dot) + (r_prop_mag * -s_dec * c_ra * self.dec_dot),
            (dr_prop_mag * c_dec * s_ra) + (r_prop_mag * c_dec *  c_ra * self.ra_dot) + (r_prop_mag * -s_dec * s_ra * self.dec_dot),
            (dr_prop_mag * s_dec       ) +                                              (r_prop_mag *  c_dec        * self.dec_dot),
        ])
        # account for Earth's rotation in velocity vector
        if assume_earth_rotation:
            # Earth's angular velocity vector in ECI
            omega_earth = np.array([0, 0, Constants.W_EARTH])
            # add velocity due to Earth's rotation
            v_veh_eci += np.cross(omega_earth, r_veh_eci)
        # generate the measured state
        measured_state = EciState(
            self.time,
            *r_veh_eci,
            *v_veh_eci,
            P=self._calcMeasuredEciStateCovariance(r_prop_mag, v_prop, assume_earth_rotation),
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
            propagated_state = propagateEciState(curr_state, dt)
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

def inverseKeplerEquation(eci_curr: EciState, dt: float) -> EciState:
    '''
    Use numerical approximation of the Inverse Kepler Equation
    (with Newton's method) to solve for the propagated ECI state
    after some time in seconds.
    Source: https://en.wikipedia.org/wiki/Kepler%27s_equation#Newton's_method

    ### Inputs:
    eci_curr (EciState) - current state in ECI
    dt (float)          - propagated time in seconds

    ### Outputs:
    (EciState) propagated state (with covariance of current state)
    '''
    # Kepler's equation to calculate eccentric and mean anomaly
    def keplersEquation(ecc, ta):
        # handle edge case for true anomaly near 180 degrees
        if abs(np.sin(ta)) < Constants.TOLERANCE and np.cos(ta) < 0:
            ecc_anom = np.pi
        # do calculation as normal
        else:
            num = np.sqrt(1 - ecc**2) * np.sin(ta)
            den = ecc + np.cos(ta)
            ecc_anom = np.arctan2(num, den)
        mean_anom = ecc_anom - ecc * np.sin(ecc_anom)
        ecc_anom = modPos(ecc_anom * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        mean_anom = modPos(mean_anom * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        return ecc_anom, mean_anom
    # get the current state's Keplerian elements
    sma, ecc, inc, raan, argp, ta = eci_curr.toKeplerianState().state
    # calculate mean motion (rad/s)
    n = np.sqrt(Constants.MU_EARTH / abs(sma**3))
    # elliptical orbits
    if ecc < 1:
        # calculate eccentric and mean anomaly
        ecc_anom, mean_anom = keplersEquation(ecc, ta)
        # propagate mean anomaly
        mean_anom_new = mean_anom + n * dt
        mean_anom_new = modPos(mean_anom_new * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        # Newton-Raphson iteration should converge in a few iterations
        ecc_anom_new = ecc_anom  # initial guess
        for _ in range(25):
            # solve for Kepler Equation root
            f = ecc_anom_new - ecc * np.sin(ecc_anom_new) - mean_anom_new
            f_prime = 1 - ecc * np.cos(ecc_anom_new)
            # update estimate
            delta_ecc_anom_new = f / f_prime
            ecc_anom_new = ecc_anom_new - delta_ecc_anom_new
            # check for convergence
            if abs(delta_ecc_anom_new) < Constants.TOLERANCE:
                break
        # convert back to true anomaly
        ta_new = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(ecc_anom_new/2), np.sqrt(1 - ecc) * np.cos(ecc_anom_new/2))
    # hyperbolic orbit (e > 1)... not going to waste my time with parabolic
    else:
        # calculate hyperbolic anomaly and mean anomaly
        if abs(np.sin(ta)) < Constants.TOLERANCE and np.cos(ta) < 0:
            hyp_anom = np.pi
        else:
            num = np.sqrt(ecc**2 - 1) * np.sin(ta)
            den = ecc + np.cos(ta)
            hyp_anom = np.arcsinh(num / den)
        # calculate mean anomaly for hyperbolic case
        mean_anom = ecc * np.sinh(hyp_anom) - hyp_anom
        # propagate mean anomaly
        mean_anom_new = mean_anom + n * dt
        # Newton-Raphson
        hyp_anom_new = hyp_anom  # Initial guess
        for _ in range(25):
            # solve for hyperbolic Kepler equation root
            f = ecc * np.sinh(hyp_anom_new) - hyp_anom_new - mean_anom_new
            f_prime = ecc * np.cosh(hyp_anom_new) - 1
            # update estimate
            delta_hyp_anom_new = f / f_prime
            hyp_anom_new = hyp_anom_new - delta_hyp_anom_new
            # check for convergence
            if abs(delta_hyp_anom_new) < Constants.TOLERANCE:
                break
        # convert back to true anomaly
        ta_new = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(hyp_anom_new / 2))
    # create the new state
    kep_new = KeplerianState(eci_curr.time + timedelta(seconds=dt), sma, ecc, inc, raan, argp, ta_new)
    eci_new = kep_new.toEciState()
    eci_new.P = eci_curr.P # reuse the current state covariance
    return eci_new

def propagateEciState(eci_curr: EciState, dt: float) -> EciState:
    '''
    Propagate the current state and its covariance forward in time.

    ### Inputs:
    eci_curr (EciState) - current state in ECI
    dt (float)          - propagated time in seconds

    ### Outputs:
    (EciState) propagated state and covariance
    '''
    # use Kepler's equation to propagate the state
    eci_new = inverseKeplerEquation(eci_curr, dt)
    # TODO: state transition matrix (linearization of dynamics)
    Phi = np.eye(6)
    # process noise (model uncertainty) which was characterized in
    # unit testing KeplerianState.toEciState
    Q = np.diag([
        5E+2, # KeplerianState -> r_eci_x
        5E+2, # KeplerianState -> r_eci_y
        9E+2, # KeplerianState -> r_eci_z
        2E-1, # KeplerianState -> v_eci_x
        7E-2, # KeplerianState -> v_eci_y
        7E-1, # KeplerianState -> v_eci_z
    ])
    # propagate covariance
    P_new = Phi @ eci_curr.P @ Phi.T + Q
    eci_new.P = P_new
    return eci_new

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
    K = prior_state.P @ H.T @ np.linalg.pinv(S)
    # state update
    updated_state = prior_state.state + K @ (measured_state.state - H @ prior_state.state)
    # covariance update using Joseph form for numerical stability
    I = np.eye(6)
    P_updated = (I - K @ H) @ prior_state.P @ (I - K @ H).T + K @ measured_state.P @ K.T
    # create updated state
    return EciState(measured_state.time, *updated_state, P=P_updated)
