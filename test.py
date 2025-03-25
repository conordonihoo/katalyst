import numpy as np
import unittest as ut
from datetime import datetime, timezone

from utils.consts import Constants
from utils.ecef import *
from utils.eci import *

class TestEcef(ut.TestCase):
    '''
    Unit tests for utils/ecef.py
    '''

    # EcefState ---------------------

    def test_00_ecef_state_to_eci_state_returns_eci_state_type_when_assuming_no_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, 0, 0, 0, 0, 0, 0)
        self.assertIsInstance(ecef.toEciState(assume_earth_rotation=False), EciState)
    def test_01_ecef_state_to_eci_state_returns_eci_state_type_when_assuming_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, 0, 0, 0, 0, 0, 0)
        self.assertIsInstance(ecef.toEciState(assume_earth_rotation=True), EciState)
    def test_02_ecef_state_to_eci_state_returns_same_state_when_assuming_no_rotation(self):
        ecef = EcefState(Constants.T0_EPOCH, 0, 0, 0, 0, 0, 0)
        eci = ecef.toEciState(assume_earth_rotation=False)
        self.assertEqual(ecef.rx, eci.rx)
        self.assertEqual(ecef.ry, eci.ry)
        self.assertEqual(ecef.rz, eci.rz)
        self.assertEqual(ecef.vx, eci.vx)
        self.assertEqual(ecef.vy, eci.vy)
        self.assertEqual(ecef.vz, eci.vz)
    def test_03_ecef_state_to_eci_state_returns_expected_state_when_assuming_rotation(self):
        rv_ecef = np.array([1E6, 1E6, 1E6, 1E2, 1E2, 1E2]) # random non-zero state
        ecef = EcefState(Constants.J2000_EPOCH, *rv_ecef)
        eci = ecef.toEciState(assume_earth_rotation=True)
        # using another source to generate this expected vector
        rv_eci_expected = [
            1.16493516e+06,
            -8.01817580e+05,
            1.00000732e+06,
            1.74960938e+02,
            4.76861822e+00,
            1.00004682e+02,
        ]
        tolerance = 1e-2 # max precision copied from external source
        self.assertTrue(abs(eci.rx - rv_eci_expected[0]) < tolerance)
        self.assertTrue(abs(eci.ry - rv_eci_expected[1]) < tolerance)
        self.assertTrue(abs(eci.rz - rv_eci_expected[2]) < tolerance)
        self.assertTrue(abs(eci.vx - rv_eci_expected[3]) < tolerance)
        self.assertTrue(abs(eci.vy - rv_eci_expected[4]) < tolerance)
        self.assertTrue(abs(eci.vz - rv_eci_expected[5]) < tolerance)

class TestEci(ut.TestCase):
    '''
    Unit tests for utils/eci.py
    '''
    def test_00(self):
        self.assertTrue(True)

class TestMain(ut.TestCase):
    '''
    Unit tests for main.py
    '''
    def test_00(self):
        self.assertTrue(True)

if __name__ == '__main__':
    ut.main()
