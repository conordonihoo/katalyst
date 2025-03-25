import numpy as np
from datetime import datetime, timedelta, timezone

class Constants:

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
    # Earth's gravitational parameter (km^3/s^2)
    MU_EARTH = 3.986004418E5

    # MEASUREMENTS --------------------

    # lat, lon, alt of GS1 (deg, deg, km)
    LLA_GS1 = (-111.536, 35.097, 2.206)
    # lat, lon, alt of GS2 (deg, deg, km)
    LLA_GS2 = (-70.692, -29.016, 2.380)
    # noise covariance of GS1
    R_GS1 = np.diag([1   , 1   , 0.01  , 0.01  ])
    # noise covariance of GS2
    R_GS2 = np.diag([0.01, 0.01, 0.0001, 0.0001])
    # noise covariance of GPS
    R_GPS = np.diag([25E6, 25E6, 25E6, 25E-2, 25E-2, 25E-2])

    # MISC. ---------------------------

    TOLERANCE = 1E-10
