from datetime import timedelta

from .consts import Constants

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
        return

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
        return
