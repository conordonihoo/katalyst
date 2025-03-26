import os
import numpy as np
import spiceypy as spice

from .consts import Constants

class KeplerianState:

    def __init__(
            self,
            sma:          float,
            ecc:          float,
            inc:          float,
            raan:         float,
            arg_perigee:  float,
            true_anomaly: float,
        ) -> None:
        '''
        The KeplerianState object initializer.

        ### Inputs:
        sma (float)          - semi-major axis (km)
        ecc (float)          - eccentricity (--)
        inc (float)          - inclination (deg)
        raan (float)         - right ascension of ascending node (deg)
        arg_perigee (float)  - argument of perigee (deg)
        true_anomaly (float) - true anomaly (deg)

        ### Outputs:
        None
        '''
        self.sma          = sma
        self.ecc          = ecc
        self.inc          = inc
        self.raan         = raan
        self.arg_perigee  = arg_perigee
        self.true_anomaly = true_anomaly
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
               f'  semi-major axis:                     {self.sma:.3f} (km)\n' \
               f'  eccentricity:                        {self.ecc:.3f} (--)\n' \
               f'  inclination:                         {self.inc:.3f} (deg)\n' \
               f'  right ascension of ascending node:   {self.raan:.3f} (deg)\n' \
               f'  argument of perigee:                 {self.arg_perigee:.3f} (deg)\n' \
               f'  true anomaly:                        {self.true_anomaly:.3f} (deg)\n' \
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
        return np.array([self.sma, self.ecc, self.inc, self.raan, self.arg_perigee, self.true_anomaly])

class EciState:

    def __init__(
            self,
            rx: float,
            ry: float,
            rz: float,
            vx: float,
            vy: float,
            vz: float,
        ) -> None:
        '''
        The EciState object initializer.

        ### Inputs:
        rx (float)   - x position in ECI (km)
        ry (float)   - y position in ECI (km)
        rz (float)   - z position in ECI (km)
        vx (float)   - x velocity in ECI (km/s)
        vy (float)   - y velocity in ECI (km/s)
        vz (float)   - z velocity in ECI (km/s)

        ### Outputs:
        None
        '''
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.vx = vx
        self.vy = vy
        self.vz = vz
        # zero covariance matrix
        self.P = np.zeros((6, 6))
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
               f'  x position:   {self.rx:.3f} (km)\n' \
               f'  y position:   {self.ry:.3f} (km)\n' \
               f'  z position:   {self.rz:.3f} (km)\n' \
               f'  x velocity:   {self.vx:.3f} (km/s)\n' \
               f'  y velocity:   {self.vy:.3f} (km/s)\n' \
               f'  z velocity:   {self.vz:.3f} (km/s)\n' \
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
        r = np.array(self.state[0:3])
        v = np.array(self.state[3:6])
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
        inc = np.rad2deg(inc)
        # right ascension of ascending node
        if n_mag < Constants.TOLERANCE: # near-equatorial orbit case
            raan = 0
        else:
            raan = np.arccos(n_vec[0] / n_mag)
            if n_vec[1] < 0:
                raan = 2 * np.pi - raan
        raan = np.rad2deg(raan)
        # argument of perigee
        if n_mag < Constants.TOLERANCE: # near-equatorial orbit
            arg_perigee = np.arctan2(e_vec[1], e_vec[0])
        elif ecc < Constants.TOLERANCE: # near-circular orbit
            arg_perigee = 0.0
        else:
            arg_perigee = np.arccos(np.dot(n_vec, e_vec) / (n_mag * ecc))
            if e_vec[2] < 0:
                arg_perigee = 2 * np.pi - arg_perigee
        arg_perigee = np.rad2deg(arg_perigee)
        # true anomaly
        if ecc < Constants.TOLERANCE: # near-circular orbit
            # use the angle between node vector and position vector
            true_anomaly = np.arccos(np.dot(n_vec, r) / (n_mag * r_mag))
            if np.dot(n_vec, v) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
        else:
            true_anomaly = np.arccos(np.dot(e_vec, r) / (ecc * r_mag))
            if np.dot(r, v) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
        true_anomaly = np.rad2deg(true_anomaly)
        # create and return KeplerianState object
        return KeplerianState(sma, ecc, inc, raan, arg_perigee, true_anomaly)

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
