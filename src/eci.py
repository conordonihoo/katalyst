import os
import numpy as np
import spiceypy as spice
from datetime import datetime

from .utils import modPos, Constants

class KeplerianState:

    def __init__(
            self,
            time: datetime,
            sma:  float,
            ecc:  float,
            inc:  float,
            raan: float,
            argp: float,
            ta:   float,
        ) -> None:
        '''
        The KeplerianState object initializer.

        ### Inputs:
        time (datetime) - time of state (--)
        sma (float)     - semi-major axis (m)
        ecc (float)     - eccentricity (--)
        inc (float)     - inclination (rad)
        raan (float)    - right ascension of ascending node (rad)
        argp (float)    - argument of perigee (rad)
        ta (float)      - true anomaly (rad)

        ### Outputs:
        None
        '''
        self.time = time
        self.sma = sma
        self.ecc = ecc
        # make sure angles are in the correct range
        self.inc = modPos(inc * Constants.RAD2DEG, 180) * Constants.DEG2RAD
        self.raan = modPos(raan * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        self.argp = modPos(argp * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        self.ta = modPos(ta * Constants.RAD2DEG, 360) * Constants.DEG2RAD
        return

    def __str__(self) -> str:
        '''
        Return this object in string format.

        ### Inputs:
        None

        ### Outputs:
        (str) representing this object
        '''
        return f'KeplerianState(\n' \
               f'  time:                 {self.time}\n' \
               f'  semi-major axis:      {self.sma / 1000:.3f} (km)\n' \
               f'  eccentricity:         {self.ecc:.3f} (--)\n' \
               f'  inclination:          {self.inc * Constants.RAD2DEG:.3f} (deg)\n' \
               f'  RAAN:                 {self.raan * Constants.RAD2DEG:.3f} (deg)\n' \
               f'  argument of perigee:  {self.argp * Constants.RAD2DEG:.3f} (deg)\n' \
               f'  true anomaly:         {self.ta * Constants.RAD2DEG:.3f} (deg)\n' \
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
        return np.array([self.sma, self.ecc, self.inc, self.raan, self.argp, self.ta])

    def toEciState(self):
        '''
        Creates an equivalent ECI state from this Keplerian state.

        ### Inputs:
        None

        ### Outputs:
        (EciState) equivalent to this Keplerian state
        '''
        # semi-latus rectum
        p = self.sma * (1 - self.ecc**2)
        # radial distance
        r_mag = p / (1 + self.ecc * np.cos(self.ta))
        # position and velocity in perifocal coords
        r_pqw = r_mag * np.array([
            np.cos(self.ta),
            np.sin(self.ta),
                          0,
        ])
        v_pqw = np.sqrt(Constants.MU_EARTH / p) * np.array([
                      -np.sin(self.ta),
            self.ecc + np.cos(self.ta),
                                     0,
        ])
        # rotate around z-axis by argp
        Rz_argp = np.array([
            [np.cos(self.argp), -np.sin(self.argp), 0],
            [np.sin(self.argp),  np.cos(self.argp), 0],
            [                0,                  0, 1],
        ])
        # rotate around x-axis by inc
        Rx_inc = np.array([
            [1,                 0,                0],
            [0, np.cos(self.inc), -np.sin(self.inc)],
            [0, np.sin(self.inc),  np.cos(self.inc)],
        ])
        # rotate around z-axis by raan
        Rz_raan = np.array([
            [np.cos(self.raan), -np.sin(self.raan), 0],
            [np.sin(self.raan),  np.cos(self.raan), 0],
            [                0,                  0, 1],
        ])
        # PQW to ECI transformation
        T_pqw_to_eci = Rz_raan @ Rx_inc @ Rz_argp
        r_eci = T_pqw_to_eci @ r_pqw
        v_eci = T_pqw_to_eci @ v_pqw
        return EciState(self.time, *r_eci, *v_eci)

class EciState:

    def __init__(
            self,
            time: datetime,
            rx:   float,
            ry:   float,
            rz:   float,
            vx:   float,
            vy:   float,
            vz:   float,
            P:    np.ndarray=np.zeros((6, 6)),
        ) -> None:
        '''
        The EciState object initializer.

        ### Inputs:
        time (datetime) - time of state (--)
        rx (float)      - x position in ECI (m)
        ry (float)      - y position in ECI (m)
        rz (float)      - z position in ECI (m)
        vx (float)      - x velocity in ECI (m/s)
        vy (float)      - y velocity in ECI (m/s)
        vz (float)      - z velocity in ECI (m/s)
        P (np.ndarray)  - state covariance matrix

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
        # state covariance matrix
        self.P = P
        return

    def __str__(self) -> str:
        '''
        Return this object in string format.

        ### Inputs:
        None

        ### Outputs:
        (str) representing this object
        '''
        covariance = str(self.P).replace('\n', '\n' + ' '*16)
        return f'EciState(\n' \
               f'  time:         {self.time}\n' \
               f'  x position:   {self.rx / 1000:.3f} (km)\n' \
               f'  y position:   {self.ry / 1000:.3f} (km)\n' \
               f'  z position:   {self.rz / 1000:.3f} (km)\n' \
               f'  x velocity:   {self.vx / 1000:.3f} (km/s)\n' \
               f'  y velocity:   {self.vy / 1000:.3f} (km/s)\n' \
               f'  z velocity:   {self.vz / 1000:.3f} (km/s)\n' \
               f'  covariance:   {covariance}\n' \
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

    def toKeplerianState(self) -> KeplerianState:
        '''
        Creates an equivalent Keplerian state from this ECI state.

        ### Inputs:
        None

        ### Outputs:
        (KeplerianState) equivalent to this ECI state
        '''
        # position and velocity vectors
        r = self.state[0:3]
        v = self.state[3:6]
        # position and velocity magnitudes
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        # specific angular momentum vector
        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)
        # node vector (perpendicular to angular momentum and z-axis)
        z_axis = np.array([0, 0, 1])
        n_vec = np.cross(z_axis, h_vec)
        n_mag = np.linalg.norm(n_vec)
        # eccentricity vector
        e_vec = np.cross(v, h_vec)/Constants.MU_EARTH - r/r_mag
        ecc = np.linalg.norm(e_vec)
        # specific orbital energy
        energy = v_mag**2/2 - Constants.MU_EARTH/r_mag
        # semi-major axis
        if abs(ecc - 1.0) < Constants.TOLERANCE: # parabolic orbit case
            sma = float('inf')
        else:
            sma = -Constants.MU_EARTH / (2 * energy)
        # inclination
        inc = np.arccos(h_vec[2] / h_mag)
        # right ascension of ascending node
        if n_mag < Constants.TOLERANCE: # near-equatorial orbit case
            raan = 0
        else:
            raan = np.arccos(n_vec[0] / n_mag)
            if n_vec[1] < 0:
                raan = 2 * np.pi - raan
        # argument of perigee
        if n_mag < Constants.TOLERANCE: # near-equatorial orbit
            argp = np.arctan2(e_vec[1], e_vec[0])
        elif ecc < Constants.TOLERANCE: # near-circular orbit
            argp = 0.0
        else:
            argp = np.arccos(np.dot(n_vec, e_vec) / (n_mag * ecc))
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp
        # true anomaly
        if ecc < Constants.TOLERANCE: # near-circular orbit
            # use the angle between node vector and position vector
            ta = np.arccos(np.dot(n_vec, r) / (n_mag * r_mag))
            if np.dot(n_vec, v) < 0:
                ta = 2 * np.pi - ta
        else:
            ta = np.arccos(np.dot(e_vec, r) / (ecc * r_mag))
            if np.dot(r, v) < 0:
                ta = 2 * np.pi - ta
        # create and return KeplerianState object
        return KeplerianState(self.time, sma, ecc, inc, raan, argp, ta)

# create global var to keep track of kernel loading
loaded_kernels = False

def loadKernels(force_load: bool=False) -> int:
    '''
    Load spiceypy kernels for ECI calculations.

    ### Inputs:
    force_load (bool) - flag to force kernel loading

    ### Outputs:
    (int) 0 if kernels were loaded, 1 otherwise
    '''
    # check if previously loaded kernels
    global loaded_kernels
    if loaded_kernels and not force_load:
        return 1
    # load kernels
    spice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    spice.furnsh(os.path.join(spice_dir, 'naif0012.tls'))
    spice.furnsh(os.path.join(spice_dir, 'earth_latest_high_prec.bpc'))
    loaded_kernels = True
    return 0
