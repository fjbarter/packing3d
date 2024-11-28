# This module contains general utilities required by functions in the
# calculator

# is_inside_boundaries and is_outside_boundaries are sufficiently fast, as
# the string comparison is only performed once, and the inside and outside
# masks are evaluated for the entire particle dataset in one go.

import numpy as np

# Relative imports
from .geometry import angular_difference

def compute_automatic_boundaries(x_data=None, y_data=None, z_data=None,
                                 r_data=None, theta_data=None,
                                 system="cartesian", padding_factor=0.1):
    """
    Compute the minimum and maximum boundaries for Cartesian or cylindrical systems.

    Args:
        x_data (np.ndarray, optional): x-coordinates of the particles (Cartesian).
        y_data (np.ndarray, optional): y-coordinates of the particles (Cartesian).
        z_data (np.ndarray): z-coordinates of the particles.
        r_data (np.ndarray, optional): Radial positions of the particles (Cylindrical).
        theta_data (np.ndarray, optional): Angular positions of the particles (Cylindrical).
        system (str): "cartesian" or "cylindrical".
        padding_factor (float): Fraction to shrink or expand the boundaries.

    Returns:
        dict: Dictionary of boundaries for the specified coordinate system.
    """
    if system == "cartesian":
        if x_data is None or y_data is None or z_data is None:
            raise ValueError("x_data, y_data, and z_data are required for Cartesian boundaries.")
        
        x_range, y_range, z_range = np.ptp(x_data), np.ptp(y_data), np.ptp(z_data)
        boundaries = {
            "x_min": min(x_data) + padding_factor * x_range,
            "x_max": max(x_data) - padding_factor * x_range,
            "y_min": min(y_data) + padding_factor * y_range,
            "y_max": max(y_data) - padding_factor * y_range,
            "z_min": min(z_data) + padding_factor * z_range,
            "z_max": max(z_data) - padding_factor * z_range,
        }
        return boundaries

    elif system == "cylindrical":
        if r_data is None or z_data is None:
            raise ValueError("r_data and z_data are required for Cylindrical boundaries.")
        
        z_range = np.ptp(z_data)
        boundaries = {
            "r_max": max(r_data) * (1 - padding_factor),
            "r_min": -max(r_data) * (1 - padding_factor),
            "theta_min": -np.pi,
            "theta_max": 3 * np.pi,
            "z_min": min(z_data) + padding_factor * z_range,
            "z_max": max(z_data) - padding_factor * z_range,
        }
        return boundaries

    else:
        raise ValueError("Invalid system specified. Choose 'cartesian' or 'cylindrical'.")


def is_inside_boundaries(x_data=None, y_data=None, z_data=None,
                         r_data=None, theta_data=None,
                         boundaries=None, radii=None,
                         factor=None, system="cartesian"):
    """
    Determine which particles are completely inside the defined boundaries
    for either Cartesian or cylindrical systems.

    Args:
        x_data (np.ndarray, optional): x-coordinates of the particles (Cartesian).
        y_data (np.ndarray, optional): y-coordinates of the particles (Cartesian).
        z_data (np.ndarray): z-coordinates of the particles.
        r_data (np.ndarray, optional): Radial distances of the particles (Cylindrical).
        theta_data (np.ndarray, optional): Angular positions of the particles (Cylindrical).
        boundaries (dict): Dictionary defining the boundaries for the system.
        radii (np.ndarray): Radii of the particles.
        factor (float, optional): Adjustment factor for angular overlaps (Cylindrical).
        system (str): "cartesian" or "cylindrical".

    Returns:
        np.ndarray: Boolean mask indicating whether each particle is inside the boundaries.
    """
    if system == "cartesian":
        if x_data is None or y_data is None or z_data is None:
            raise ValueError("x_data, y_data, and z_data are required for Cartesian boundaries.")
        
        return (
            (x_data >= boundaries["x_min"] + radii) &
            (x_data <= boundaries["x_max"] - radii) &
            (y_data >= boundaries["y_min"] + radii) &
            (y_data <= boundaries["y_max"] - radii) &
            (z_data >= boundaries["z_min"] + radii) &
            (z_data <= boundaries["z_max"] - radii)
        )

    elif system == "cylindrical":
        if r_data is None or z_data is None or theta_data is None:
            raise ValueError("r_data, theta_data, and z_data are required for Cylindrical boundaries.")

        # Full cylinder
        if boundaries["r_min"] < 0:
            return (
                (r_data <= boundaries["r_max"] - radii) &
                (z_data >= boundaries["z_min"] + radii) &
                (z_data <= boundaries["z_max"] - radii)
            )

        # Full ring
        if boundaries["theta_min"] == 0 and boundaries["theta_max"] == 2 * np.pi:
            return (
                (r_data >= boundaries["r_min"] + radii) &
                (r_data <= boundaries["r_max"] - radii) &
                (z_data >= boundaries["z_min"] + radii) &
                (z_data <= boundaries["z_max"] - radii)
            )

        # Theta range handling
        if factor is None:
            raise ValueError("factor is required for Cylindrical boundaries with angular constraints.")

        theta_min = (boundaries["theta_min"] + factor) % (2 * np.pi)
        theta_max = (boundaries["theta_max"] - factor) % (2 * np.pi)

        # Theta condition for periodic ranges
        standard_range = (theta_min <= theta_max)
        theta_inside = np.where(
            standard_range,
            (theta_data >= theta_min) & (theta_data <= theta_max),
            (theta_data >= theta_min) | (theta_data <= theta_max)
        )

        return (
            (r_data >= boundaries["r_min"] + radii) &
            (r_data <= boundaries["r_max"] - radii) &
            theta_inside &
            (z_data >= boundaries["z_min"] + radii) &
            (z_data <= boundaries["z_max"] - radii)
        )

    else:
        raise ValueError("Invalid system specified. Choose 'cartesian' or 'cylindrical'.")


def is_outside_boundaries(x_data=None, y_data=None, z_data=None,
                          r_data=None, theta_data=None,
                          boundaries=None, radii=None,
                          factor=None, system="cartesian"):
    """
    Determine which particles are completely outside the defined boundaries
    for either Cartesian or cylindrical systems.

    Args:
        x_data (np.ndarray, optional): x-coordinates of the particles (Cartesian).
        y_data (np.ndarray, optional): y-coordinates of the particles (Cartesian).
        z_data (np.ndarray): z-coordinates of the particles.
        r_data (np.ndarray, optional): Radial distances of the particles (Cylindrical).
        theta_data (np.ndarray, optional): Angular positions of the particles (Cylindrical).
        boundaries (dict): Dictionary defining the boundaries for the system.
        radii (np.ndarray): Radii of the particles.
        factor (float, optional): Adjustment factor for angular overlaps (Cylindrical).
        system (str): "cartesian" or "cylindrical".

    Returns:
        np.ndarray: Boolean mask indicating whether each particle is outside the boundaries.
    """
    if system == "cartesian":
        if x_data is None or y_data is None or z_data is None:
            raise ValueError("x_data, y_data, and z_data are required for Cartesian boundaries.")
        
        return (
            (x_data <= boundaries["x_min"] - radii) |
            (x_data >= boundaries["x_max"] + radii) |
            (y_data <= boundaries["y_min"] - radii) |
            (y_data >= boundaries["y_max"] + radii) |
            (z_data <= boundaries["z_min"] - radii) |
            (z_data >= boundaries["z_max"] + radii)
        )

    elif system == "cylindrical":
        if r_data is None or z_data is None or theta_data is None:
            raise ValueError("r_data, theta_data, and z_data are required for Cylindrical boundaries.")

        # Full cylinder
        if boundaries["r_min"] < 0:
            return (
                (r_data >= boundaries["r_max"] + radii) |
                (z_data <= boundaries["z_min"] - radii) |
                (z_data >= boundaries["z_max"] + radii)
            )

        # Full ring
        if boundaries["theta_min"] == 0 and boundaries["theta_max"] == 2 * np.pi:
            return (
                (r_data <= boundaries["r_min"] - radii) |
                (r_data >= boundaries["r_max"] + radii) |
                (z_data <= boundaries["z_min"] - radii) |
                (z_data >= boundaries["z_max"] + radii)
            )

        # Theta range handling
        if factor is None:
            raise ValueError("factor is required for Cylindrical boundaries with angular constraints.")

        theta_min = (boundaries["theta_min"] - factor) % (2 * np.pi)
        theta_max = (boundaries["theta_max"] + factor) % (2 * np.pi)

        # Theta condition for periodic ranges (outside check)
        standard_range = (theta_min <= theta_max)
        theta_outside = np.where(
            standard_range,
            (theta_data <= theta_min) | (theta_data >= theta_max),
            (theta_data <= theta_min) & (theta_data >= theta_max)
        )

        return (
            (r_data <= boundaries["r_min"] - radii) |
            (r_data >= boundaries["r_max"] + radii) |
            theta_outside |
            (z_data <= boundaries["z_min"] - radii) |
            (z_data >= boundaries["z_max"] + radii)
        )

    else:
        raise ValueError("Invalid system specified. Choose 'cartesian' or 'cylindrical'.")


def calculate_overlaps(x_data=None, y_data=None, z_data=None,
                       r_data=None, theta_data=None,
                       radii=None, boundaries=None,
                       factor=None, system="cartesian"):
    """
    Calculate boolean masks for particles overlapping with each boundary
    for either Cartesian or cylindrical systems.

    Args:
        x_data, y_data, z_data (np.ndarray, optional): Cartesian coordinates of particles.
        r_data, theta_data, z_data (np.ndarray, optional): Cylindrical coordinates of particles.
        radii (np.ndarray): Array of particle radii.
        boundaries (dict): Dictionary defining the boundary limits.
        factor (float, optional): Buffer added to boundaries for overlap checks (Cylindrical).
        system (str): "cartesian" or "cylindrical".

    Returns:
        dict: Boolean masks for overlaps with each boundary ('x_min', 'r_min', etc.).
    """
    if system == "cartesian":
        if x_data is None or y_data is None or z_data is None:
            raise ValueError("x_data, y_data, and z_data are required for Cartesian overlaps.")
        
        overlaps = {
            "x_min": (x_data > boundaries["x_min"] - radii) & 
                     (x_data < boundaries["x_min"] + radii),
            "x_max": (x_data > boundaries["x_max"] - radii) & 
                     (x_data < boundaries["x_max"] + radii),
            "y_min": (y_data > boundaries["y_min"] - radii) & 
                     (y_data < boundaries["y_min"] + radii),
            "y_max": (y_data > boundaries["y_max"] - radii) & 
                     (y_data < boundaries["y_max"] + radii),
            "z_min": (z_data > boundaries["z_min"] - radii) & 
                     (z_data < boundaries["z_min"] + radii),
            "z_max": (z_data > boundaries["z_max"] - radii) & 
                     (z_data < boundaries["z_max"] + radii),
        }
        return overlaps

    elif system == "cylindrical":
        if r_data is None or theta_data is None or z_data is None:
            raise ValueError("r_data, theta_data, and z_data are required for Cylindrical overlaps.")
        
        def theta_overlap(theta, theta_min, theta_max):
            """
            Check if theta values overlap with a periodic range [theta_min, theta_max).

            Args:
                theta (np.ndarray): Array of theta values.
                theta_min (np.ndarray or float): Minimum bound of the range.
                theta_max (np.ndarray or float): Maximum bound of the range.

            Returns:
                np.ndarray: Boolean mask indicating overlap for each theta value.
            """
            theta_min = np.asarray(theta_min)
            theta_max = np.asarray(theta_max)

            standard_mask = (theta_min <= theta_max) & (theta > theta_min) & (theta < theta_max)
            wrapped_mask = (theta_min > theta_max) & ((theta > theta_min) | (theta < theta_max))

            return standard_mask | wrapped_mask

        # Handle periodicity in theta
        theta_min_wrapped = (boundaries["theta_min"] - factor) % (2 * np.pi)
        theta_max_wrapped = (boundaries["theta_max"] + factor) % (2 * np.pi)

        if boundaries["r_min"] < 0:  # Full cylindrical region -> no theta overlap
            return {
                "r_min": False,
                "r_max": (r_data > boundaries["r_max"] - radii) & 
                         (r_data < boundaries["r_max"] + radii),
                "theta_min": False,
                "theta_max": False,
                "z_min": (z_data > boundaries["z_min"] - radii) & 
                         (z_data < boundaries["z_min"] + radii),
                "z_max": (z_data > boundaries["z_max"] - radii) & 
                         (z_data < boundaries["z_max"] + radii),
            }

        overlaps = {
            "r_min": (r_data > boundaries["r_min"] - radii) & 
                     (r_data < boundaries["r_min"] + radii),
            "r_max": (r_data > boundaries["r_max"] - radii) & 
                     (r_data < boundaries["r_max"] + radii),
            "theta_min": theta_overlap(theta_data, theta_min_wrapped, boundaries["theta_min"] + factor),
            "theta_max": theta_overlap(theta_data, boundaries["theta_max"] - factor, theta_max_wrapped),
            "z_min": (z_data > boundaries["z_min"] - radii) & 
                     (z_data < boundaries["z_min"] + radii),
            "z_max": (z_data > boundaries["z_max"] - radii) & 
                     (z_data < boundaries["z_max"] + radii),
        }
        return overlaps

    else:
        raise ValueError("Invalid system specified. Choose 'cartesian' or 'cylindrical'.")


def calculate_active_overlap_values(total_particles, x_data=None, y_data=None, z_data=None,
                                    r_data=None, theta_data=None,
                                    boundaries=None, overlaps=None, system="cartesian"):
    """
    Calculate the overlap distances for particles intersecting the boundaries
    for either Cartesian or cylindrical systems.

    Args:
        total_particles (int): Total number of particles.
        x_data, y_data, z_data (np.ndarray, optional): Cartesian coordinates of particles.
        r_data, theta_data, z_data (np.ndarray, optional): Cylindrical coordinates of particles.
        boundaries (dict): Dictionary defining the boundaries for the system.
        overlaps (dict): Dictionary of boolean masks indicating overlaps with boundaries.
        system (str): "cartesian" or "cylindrical".

    Returns:
        np.ndarray: Array of distances between particle centers and boundaries, or NaN for no overlap.
    """
    # Initialize the output array
    active_overlap_values = np.full((total_particles, 6), np.nan, dtype=float)

    if system == "cartesian":
        if x_data is None or y_data is None or z_data is None:
            raise ValueError("x_data, y_data, and z_data are required for Cartesian overlaps.")
        
        # Cartesian overlaps
        active_overlap_values[:, 0] = np.where(
            overlaps["x_min"], boundaries["x_min"] - x_data, np.nan)
        active_overlap_values[:, 1] = np.where(
            overlaps["x_max"], x_data - boundaries["x_max"], np.nan)
        active_overlap_values[:, 2] = np.where(
            overlaps["y_min"], boundaries["y_min"] - y_data, np.nan)
        active_overlap_values[:, 3] = np.where(
            overlaps["y_max"], y_data - boundaries["y_max"], np.nan)
        active_overlap_values[:, 4] = np.where(
            overlaps["z_min"], boundaries["z_min"] - z_data, np.nan)
        active_overlap_values[:, 5] = np.where(
            overlaps["z_max"], z_data - boundaries["z_max"], np.nan)

    elif system == "cylindrical":
        if r_data is None or theta_data is None or z_data is None:
            raise ValueError("r_data, theta_data, and z_data are required for Cylindrical overlaps.")
        
        # Cylindrical overlaps
        # Radial overlaps
        active_overlap_values[:, 0] = np.where(
            overlaps["r_min"], boundaries["r_min"] - r_data, np.nan)
        active_overlap_values[:, 1] = np.where(
            overlaps["r_max"], r_data - boundaries["r_max"], np.nan)

        # Angular overlaps
        theta_min_differences = angular_difference(theta_data, boundaries["theta_min"])
        theta_max_differences = angular_difference(theta_data, boundaries["theta_max"])

        # Convert angular differences to distances
        theta_min_distances = r_data * np.sin(theta_min_differences)
        theta_max_distances = r_data * np.sin(theta_max_differences)

        active_overlap_values[:, 2] = np.where(overlaps["theta_min"], theta_min_distances, np.nan)
        active_overlap_values[:, 3] = np.where(overlaps["theta_max"], theta_max_distances, np.nan)

        # z overlaps
        active_overlap_values[:, 4] = np.where(
            overlaps["z_min"], boundaries["z_min"] - z_data, np.nan)
        active_overlap_values[:, 5] = np.where(
            overlaps["z_max"], z_data - boundaries["z_max"], np.nan)

    else:
        raise ValueError("Invalid system specified. Choose 'cartesian' or 'cylindrical'.")

    return active_overlap_values


