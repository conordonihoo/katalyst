import numpy as np
from datetime import datetime, timedelta, timezone

class Constants:
    '''
    Class to store all constants used in this project.
    Because there is no `self`, this can be treated like
    a C++ namespace.
    '''

    # TIME ----------------------------

    # J2000 Julian date
    J2000_JD = 2451545
    # J2000 epoch
    J2000_EPOCH = datetime(2000, 1, 1, 11, 58, 55, microsecond=816000, tzinfo=timezone.utc)
    # To epoch
    T0_EPOCH = datetime(2024, 11, 24, 5, 4, 30, tzinfo=timezone.utc)
    # Tf epoch
    TF_EPOCH = T0_EPOCH + timedelta(seconds=12000)
    # Tf+5 epoch
    T5_EPOCH = TF_EPOCH + timedelta(hours=5)

    # EARTH ---------------------------

    # Earth's radius (km)
    R_EARTH = 6378
    # Earth's rotation rate, 15 deg/hr, (rad/s)
    W_EARTH = np.deg2rad(15 / 3600)
    # Earth's gravitational parameter (km^3/s^2)
    MU_EARTH = 3.986004418E5

    # MEASUREMENTS --------------------

    # ID for GS1
    ID_GS1 = 'ground_observer_1'
    # ID for GS2
    ID_GS2 = 'ground_observer_2'
    # ID for GPS
    ID_GPS = 'gps_measurement'
    # lat, lon, alt of GS1 (deg, deg, km)
    LLA_GS1 = (-111.536, 35.097, 2.206)
    # lat, lon, alt of GS2 (deg, deg, km)
    LLA_GS2 = (-70.692, -29.016, 2.380)
    # noise covariance of GS1
    # TODO: make sure units are correct for applications
    R_GS1 = np.diag([1   , 1   , 0.01  , 0.01  ])
    # noise covariance of GS2
    # TODO: make sure units are correct for applications
    R_GS2 = np.diag([0.01, 0.01, 0.0001, 0.0001])
    # noise covariance of GPS
    # TODO: make sure units are correct for applications
    R_GPS = np.diag([25E6, 25E6, 25E6, 25E-2, 25E-2, 25E-2])

    # MISC. ---------------------------

    TOLERANCE = 1E-10
