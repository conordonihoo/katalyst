# Orbital State Estimation Project

This project implements a satellite state estimation system using various coordinate frames, state representations, and sensor measurements. It combines GPS measurements and ground-based observations to determine and track the state of an orbiting satellite.

---

## Core Concepts

### Coordinate Frames
- **Earth-Centered Earth-Fixed (ECEF)**: Coordinate system that rotates with the Earth
- **Earth-Centered Inertial (ECI)**: Non-rotating coordinate system fixed relative to distant stars
- **Keplerian Elements**: Orbital parameters describing a satellite's orbit

### State Representations
- `EcefState`: Position and velocity in ECEF coordinates
- `EciState`: Position and velocity in ECI coordinates
- `KeplerianState`: Orbital elements (semi-major axis, eccentricity, inclination, RAAN, argument of perigee, true anomaly)

### Measurements
- `GpsMeasurement`: GPS-derived position and velocity in ECEF
- `GroundStationMeasurement`: Right ascension (RA) and declination (Dec) measurements from ground stations

---

## Project Structure

### `src/utils.py`
Contains utility functions and constants used throughout the project:
- `Constants`: Earth parameters, conversion factors, measurement parameters
- `modPos()`: Function to ensure angles are in the proper range

### `src/ecef.py`
Handles Earth-Centered Earth-Fixed coordinate system:
- `EcefState`: Class for representing position and velocity in ECEF
- `generateEcefToEciTransform()`: Generates transformation matrix from ECEF to ECI
- `geodeticToEcef()`: Converts latitude, longitude, altitude to ECEF coordinates

### `src/eci.py`
Handles Earth-Centered Inertial coordinate system:
- `EciState`: Class for representing position and velocity in ECI
- `KeplerianState`: Class for representing orbital elements
- `loadKernels()`: Loads SPICE kernels for coordinate transformations

### `src/nav.py`
Contains the main navigation and estimation algorithms:
- `GpsMeasurement`: Class for GPS measurements
- `GroundStationMeasurement`: Class for ground station measurements
- `measurementsToEciStates()`: Converts measurements to ECI states
- `kalmanUpdate()`: Performs Kalman filter update step
- `propagateEciState()`: Propagates state forward in time using orbital mechanics
- `inverseKeplerEquation()`: Solves Kepler's equation for orbit propagation

### `main.py`
Entry point for the application, parses CSV data and runs estimation:
- `parseCsv()`: Reads measurements from CSV file
- `main()`: Orchestrates the state estimation process

---

## Key Algorithms

### Coordinate Transformations
The project implements bidirectional transformations between coordinate systems:
- ECEF ↔ ECI using rotation matrices that account for Earth's rotation (if applicable)
- ECI ↔ Keplerian using orbital mechanics equations

### State Propagation
Uses Kepler's equations to propagate orbital state forward in time, accounting for orbital mechanics.

### Measurement Processing
Processes two types of measurements:
1. **GPS Measurements**:
    * Direct position/velocity measurements in ECEF, transformed to ECI
    * Measurement covariances in ECEF, transformed to ECI
2. **Ground Station Measurements**:
    * Angle measurements (RA/Dec) from ground observers, converted to ECI position/velocity using the propagated state's estimated range
    * Measurement covariances in squared radians, converted to squared meters

### Kalman Filtering
Implements a Kalman filter for optimal state estimation:
- Predicts state forward using orbital mechanics
- Updates state using measurement information
- Handles measurement and process noise covariance

---

## Usage

```bash
# Run with no Earth rotation assumption
python main.py

# Run assuming Earth rotation
python main.py --rotating-earth
```

---

## Testing

The project includes unit tests in `test.py` that verify:
- Coordinate transformations
- State conversions
- Kernel loading

---

## Dependencies

- `numpy`: For numerical operations
- `spiceypy`: For high-precision ECEF ↔ ECI transformations (assuming Earth rotation)
- Standard Python libraries: `datetime`, `argparse`, etc.
