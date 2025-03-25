import numpy as np
import unittest as ut

from src.consts import Constants
from src.ecef import *
from src.eci import *
from src.nav import *

# useful test constants
ECEF_STATE_NONZERO = np.array([
    1E6,
    1E6,
    1E6,
    1E2,
    1E2,
    1E2,
])
ECI_STATE_NONZERO = np.array([
    1.16493516e+06,
    -8.01817580e+05, # used another source to
    1.00000732e+06,  # generate this ECI state
    1.74960938e+02,  # equivalent to the non-zero
    4.76861822e+00,  # ECEF state at J2000
    1.00004682e+02,
])
ECI_STATE_VALLADO = np.array([
    6524834e-03,
    6862875e-03,     # example ECI state taken
    6448296e-03,     # from Vallado, 4th ed.
    4901327e-06,     # pg. 114
    5533756e-06,
    -1976341e-06,
])
KEPLERIAN_STATE_VALLADO = np.array([
    36127343e-03,
    832853e-06,      # example Keplerian state taken
    87.870,          # from Vallado, 4th ed.
    227.898,         # pg. 114
    53.38,
    92.335,
])

class TestEcef(ut.TestCase):
    '''
    Unit tests for src/ecef.py
    '''

    # EcefState --------------------------

    def test_00_ecef_state_to_eci_state_returns_eci_state_type_when_assuming_no_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, *ECEF_STATE_NONZERO)
        self.assertIsInstance(ecef.toEciState(assume_earth_rotation=False), EciState)
    def test_01_ecef_state_to_eci_state_returns_eci_state_type_when_assuming_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, *ECEF_STATE_NONZERO)
        self.assertIsInstance(ecef.toEciState(assume_earth_rotation=True), EciState)
    def test_02_ecef_state_to_eci_state_returns_same_state_when_assuming_no_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, *ECEF_STATE_NONZERO)
        eci = ecef.toEciState(assume_earth_rotation=False)
        self.assertEqual(ecef.rx, eci.rx)
        self.assertEqual(ecef.ry, eci.ry)
        self.assertEqual(ecef.rz, eci.rz)
        self.assertEqual(ecef.vx, eci.vx)
        self.assertEqual(ecef.vy, eci.vy)
        self.assertEqual(ecef.vz, eci.vz)
    def test_03_ecef_state_to_eci_state_returns_expected_state_when_assuming_rotation(self):
        ecef = EcefState(Constants.J2000_EPOCH, *ECEF_STATE_NONZERO)
        eci = ecef.toEciState(assume_earth_rotation=True)
        # assigning delta manually to learn the accuracy
        self.assertAlmostEqual(eci.rx, ECI_STATE_NONZERO[0], delta=3E-3)
        self.assertAlmostEqual(eci.ry, ECI_STATE_NONZERO[1], delta=4E-4)
        self.assertAlmostEqual(eci.rz, ECI_STATE_NONZERO[2], delta=3E-3)
        self.assertAlmostEqual(eci.vx, ECI_STATE_NONZERO[3], delta=2E-7)
        self.assertAlmostEqual(eci.vy, ECI_STATE_NONZERO[4], delta=2E-9)
        self.assertAlmostEqual(eci.vz, ECI_STATE_NONZERO[5], delta=5E-7)

    # geodeticToEcef ---------------------

    def test_04_geodetic_to_ecef_returns_correct_ecef_coordinates(self):
        lat = 0   # (deg)
        lon = 0   # (deg)
        alt = 100 # (km)
        self.assertEqual(geodeticToEcef(lat, lon, alt), (Constants.R_EARTH + alt, 0, 0))


class TestEci(ut.TestCase):
    '''
    Unit tests for src/eci.py
    '''

    # EciState ---------------------------

    def test_00_eci_state_to_keplerian_state_returns_keplerian_state_type(self):
        eci = EciState(*ECI_STATE_VALLADO)
        self.assertIsInstance(eci.toKeplerianState(), KeplerianState)
    def test_01_eci_state_to_keplerian_state_returns_expected_state(self):
        eci = EciState(*ECI_STATE_VALLADO)
        kep = eci.toKeplerianState()
        # assigning delta manually to learn the accuracy
        self.assertAlmostEqual(kep.sma,          KEPLERIAN_STATE_VALLADO[0], delta=6E-3)
        self.assertAlmostEqual(kep.ecc,          KEPLERIAN_STATE_VALLADO[1], delta=4E-7)
        self.assertAlmostEqual(kep.inc,          KEPLERIAN_STATE_VALLADO[2], delta=9E-4)
        self.assertAlmostEqual(kep.raan,         KEPLERIAN_STATE_VALLADO[3], delta=3E-4)
        self.assertAlmostEqual(kep.arg_perigee,  KEPLERIAN_STATE_VALLADO[4], delta=5E-3)
        self.assertAlmostEqual(kep.true_anomaly, KEPLERIAN_STATE_VALLADO[5], delta=2E-4)

    # loadKernels ------------------------

    def test_02_load_kernels_does_not_load_if_previously_loaded(self):
        loadKernels() # call once
        self.assertEqual(loadKernels(), 1)
    def test_03_load_kernels_does_load_if_forced(self):
        loadKernels() # call once
        self.assertEqual(loadKernels(force_load=True), 0)

class TestNav(ut.TestCase):
    '''
    Unit tests for src/nav.py
    '''

    # GpsMeasurement ---------------------

    def test_00_gps_measurement_time_is_set_from_t0_epoch(self):
        elapsed_seconds = 123
        gps = GpsMeasurement(elapsed_seconds, Constants.ID_GPS, *ECEF_STATE_NONZERO)
        self.assertEqual((gps.time - Constants.T0_EPOCH).total_seconds(), elapsed_seconds)

    # GroundStationMeasurement -----------

    def test_01_ground_station_measurement_time_is_set_from_t0_epoch(self):
        elapsed_seconds = 123
        gs = GroundStationMeasurement(elapsed_seconds, Constants.ID_GS1, 0, 0, 0, 0)
        self.assertEqual((gs.time - Constants.T0_EPOCH).total_seconds(), elapsed_seconds)


class TestMain(ut.TestCase):
    '''
    Unit tests for main.py
    '''
    def test_00(self):
        self.assertTrue(True)

if __name__ == '__main__':
    ut.main()
