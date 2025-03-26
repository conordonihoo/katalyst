import numpy as np
from datetime import datetime, timedelta, timezone

class Constants:
    '''
    Class to store all constants used in this project.
    Because there is no `self`, this can be treated like
    a C++ namespace.
    '''

    # MISC. ---------------------------

    ARCSEC2DEG = 1/3600
    DEG2RAD = np.pi/180
    RAD2DEG = 1/DEG2RAD
    TOLERANCE = 1E-10

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

    # Earth's radius (m)
    R_EARTH = 6378000
    # Earth's rotation rate, 15 deg/hr, (rad/s)
    W_EARTH = DEG2RAD * 15 / 3600
    # Earth's gravitational parameter (m^3/s^2)
    MU_EARTH = 3.986004418E14

    # MEASUREMENTS --------------------

    # ID for GS1
    ID_GS1 = 'ground_observer_1'
    # ID for GS2
    ID_GS2 = 'ground_observer_2'
    # ID for GPS
    ID_GPS = 'gps_measurement'
    # lat, lon, alt of GS1 (rad, rad, m)
    LLA_GS1 = (DEG2RAD * -111.536, DEG2RAD * 35.097, 2206)
    # lat, lon, alt of GS2 (rad, rad, m)
    LLA_GS2 = (DEG2RAD * -70.692, DEG2RAD * -29.016, 2380)
    # noise covariance of GS1 (rad^2) and (rad^2/s^2)
    R_GS1 = (ARCSEC2DEG * DEG2RAD)**2 * np.diag([1, 1, 1E-2, 1E-2])
    # noise covariance of GS2 (rad^2) and (rad^2/s^2)
    R_GS2 = (ARCSEC2DEG * DEG2RAD)**2 * np.diag([1E-2, 1E-2, 1E-4, 1E-4])
    # noise covariance of GPS (m^2) and (m^2/s^2)
    R_GPS = np.diag([25E6, 25E6, 25E6, 25E-2, 25E-2, 25E-2])
