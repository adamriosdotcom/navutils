import math
import numpy as np

# --------------------------------------
# 1. Define WGS84 Ellipsoid Parameters
# --------------------------------------
# Semi-major axis (meters)
WGS84_A = 6378137.0
# Flattening
WGS84_F = 1.0 / 298.257223563
# Eccentricity squared
WGS84_E2 = 2 * WGS84_F - WGS84_F**2


def llh_to_ecef(lat_deg, lon_deg, h):
    """
    Convert geodetic coordinates (latitude, longitude, height) 
    to Earth-Centered, Earth-Fixed (ECEF) coordinates (x, y, z).

    lat_deg, lon_deg in degrees
    h in meters
    Returns ECEF coordinates [x, y, z] in meters
    """
    # Convert degrees to radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Prime vertical radius of curvature
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)

    # Compute ECEF coordinates
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = ((1 - WGS84_E2) * N + h) * sin_lat

    return np.array([x, y, z])


def ecef_to_ned_matrix(lat_deg, lon_deg):
    """
    Compute the ECEF-to-NED rotation matrix given 
    a reference latitude/longitude in degrees.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Rows of the rotation matrix for NED:
    #   N-axis: -sin(lat)*cos(lon)*X  -sin(lat)*sin(lon)*Y  +cos(lat)*Z
    #   E-axis:         -sin(lon)*X          +cos(lon)*Y     0*Z
    #   D-axis: -cos(lat)*cos(lon)*X -cos(lat)*sin(lon)*Y  -sin(lat)*Z
    r11 = -sin_lat * cos_lon
    r12 = -sin_lat * sin_lon
    r13 =  cos_lat
    r21 = -sin_lon
    r22 =  cos_lon
    r23 =  0.0
    r31 = -cos_lat * cos_lon
    r32 = -cos_lat * sin_lon
    r33 = -sin_lat

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])


def llh_to_ned(lat_ref, lon_ref, h_ref, lat_point, lon_point, h_point):
    """
    Given two points in geodetic coordinates (LLH),
    return the NED coordinates of the second 'point' 
    with respect to the first 'ref' as the origin.
    """
    # 1) Convert both points to ECEF
    ecef_ref = llh_to_ecef(lat_ref, lon_ref, h_ref)
    ecef_pt  = llh_to_ecef(lat_point, lon_point, h_point)

    # 2) Compute the ECEF->NED rotation matrix at the first (origin) point
    R = ecef_to_ned_matrix(lat_ref, lon_ref)

    # 3) Compute the difference in ECEF
    diff_ecef = ecef_pt - ecef_ref

    # 4) Transform this difference into the local NED frame
    ned_coords = R.dot(diff_ecef)

    return ned_coords  # [North, East, Down]


def ecef_velocity_to_ned_velocity(lat_ref, lon_ref, vel_ecef):
    """
    Convert a velocity vector in ECEF (v_x, v_y, v_z)
    into NED given a reference lat/long for orientation.
    
    vel_ecef is a 3-element array [vx, vy, vz] (m/s).
    Returns velocity in the NED frame: [v_N, v_E, v_D].
    """
    # Build the ECEF->NED rotation matrix at the reference
    R = ecef_to_ned_matrix(lat_ref, lon_ref)
    # Rotate ECEF velocity to NED
    vel_ned = R.dot(vel_ecef)
    return vel_ned


def compute_ned_velocity_from_two_llh(
    lat_ref, lon_ref, h_ref,
    lat1, lon1, h1, t1, 
    lat2, lon2, h2, t2
):
    """
    Example function:
    Given a reference LLH (lat_ref, lon_ref, h_ref),
    and two successive LLH readings with times t1, t2,
    compute the approximate velocity in NED at the reference.

    1) Convert each LLH to ECEF.
    2) Numerical difference to find velocity in ECEF.
    3) Rotate that velocity into NED using the reference orientation.
    
    Returns v_ned = [vN, vE, vD]
    """
    # Convert both points to ECEF
    ecef1 = llh_to_ecef(lat1, lon1, h1)
    ecef2 = llh_to_ecef(lat2, lon2, h2)

    # Time difference
    dt = t2 - t1
    if abs(dt) < 1e-9:
        raise ValueError("Timestamps are the same or too close; cannot compute velocity.")

    # Approx ECEF velocity vector (m/s)
    vel_ecef = (ecef2 - ecef1) / dt

    # Convert to NED using the reference's lat/lon
    # (You could use lat1, lon1 or a midpoint latitude/longitude as well)
    vel_ned = ecef_velocity_to_ned_velocity(lat_ref, lon_ref, vel_ecef)
    return vel_ned


def main():
    # ------------------------------------
    # Example usage: Position in NED
    # ------------------------------------
    lat_ref = 37.4276
    lon_ref = -122.1697
    h_ref   = 30.0

    # Another point
    lat_2 = 37.4280
    lon_2 = -122.1700
    h_2   = 35.0

    # Position of point_2 in NED w.r.t. reference
    ned_pos = llh_to_ned(lat_ref, lon_ref, h_ref, lat_2, lon_2, h_2)
    print("NED Position of second point relative to reference (m):")
    print(f"  North: {ned_pos[0]:.3f}")
    print(f"  East:  {ned_pos[1]:.3f}")
    print(f"  Down:  {ned_pos[2]:.3f}\n")

    # ------------------------------------
    # Example usage: Velocity in NED
    # ------------------------------------
    # Suppose you have two LLH fixes at times t1, t2
    lat1, lon1, h1, t1 = 37.4276, -122.1697, 30.0, 100.0   # time in seconds
    lat2, lon2, h2, t2 = 37.4277, -122.1699, 31.0, 102.0   # 2 seconds later

    # We use the reference lat/lon (lat_ref, lon_ref) for orientation,
    # but you can choose whichever makes sense for your application
    vel_ned = compute_ned_velocity_from_two_llh(
        lat_ref, lon_ref, h_ref,
        lat1, lon1, h1, t1,
        lat2, lon2, h2, t2
    )
    print("Approximate NED Velocity (m/s) between two LLH points:")
    print(f"  vN: {vel_ned[0]:.3f}")
    print(f"  vE: {vel_ned[1]:.3f}")
    print(f"  vD: {vel_ned[2]:.3f}")


if __name__ == "__main__":
    main()
