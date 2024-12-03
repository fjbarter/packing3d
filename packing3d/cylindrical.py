# This module contains all the necessary functions that are unique to
# calculating packing density in cylindrical coordinates

import numpy as np


from .io import (read_vtk_file,
                 retrieve_coordinates)

from .geometry import (convert_to_cylindrical,
                       calculate_angular_overlap_factor,
                       compute_cell_volume,
                       single_cap_intersection,
                       double_cap_intersection,
                       triple_cap_intersection,
                       sphere_cylinder_intersection,
                       sphere_cylinder_plane_intersection)

from .utils import (compute_automatic_boundaries,
                    calculate_overlaps,
                    calculate_active_overlap_values,
                    is_inside_boundaries,
                    is_outside_boundaries)

from .cartesian import calculate_particle_volume


def compute_packing_cylindrical(file=None, boundaries=None,
                                r_data=None, theta_data=None, z_data=None,
                                radii=None, accurate_cylindrical=False):
    """
    Compute the packing density of particles within a cylindrical region.

    Args:
        file (str, optional): Path to the VTK file containing particle data.
                              Required if `r_data`, `theta_data`, `z_data`,
                              and `radii` are not provided.
        boundaries (dict, optional): Dictionary defining the cylindrical region
                                     boundaries, with keys `r_min`, `r_max`,
                                     `theta_min`, `theta_max`, `z_min`, and
                                     `z_max`.
        r_data (np.ndarray, optional): Radial coordinates of the particles.
        theta_data (np.ndarray, optional): Angular coordinates of particles.
        z_data (np.ndarray, optional): Z-coordinates of the particles.
        radii (np.ndarray, optional): Radii of the particles.
        accurate_cylindrical (bool, optional): If True, computes accurate
                                               cylindrical overlaps. Otherwise,
                                               approximates overlaps as planar.

    Returns:
        float: The packing density as the fraction of the cylindrical volume
               occupied by particles.

    Raises:
        ValueError: If neither `file` nor particle data (`r_data`,
                    `theta_data`, `z_data`, `radii`) are provided.
    """
    if r_data is None or theta_data is None or z_data is None or radii is None:
        if file is None:
            raise ValueError("Either 'file' or data must be provided.")
        try:
            data = read_vtk_file(file)
            # Retrieve particle coordinates and radii
            x_data, y_data, z_data, radii = retrieve_coordinates(data)
            r_data, theta_data = convert_to_cylindrical(x_data, y_data)
        except FileNotFoundError as fnf_error:
            print(fnf_error)
        except ValueError as value_error:
            print(value_error)

    # Determine boundaries
    if boundaries is None:
        boundaries = compute_automatic_boundaries(x_data=None, y_data=None,
                                                  r_data=r_data,
                                                  theta_data=theta_data,
                                                  z_data=z_data,
                                                  system="cylindrical")

    # Calculate angular overlap factor
    factor = calculate_angular_overlap_factor(r_data, radii)

    # Calculate overlaps
    overlaps = calculate_overlaps(x_data=None, y_data=None,
                                  r_data=r_data, theta_data=theta_data,
                                  z_data=z_data, radii=radii,
                                  boundaries=boundaries,
                                  factor=factor,
                                  system="cylindrical")

    # Determine active overlap values
    total_particles = len(radii)
    active_overlap_values = (
        calculate_active_overlap_values(total_particles,
                                        x_data=None,y_data=None,
                                        r_data=r_data,theta_data=theta_data,
                                        z_data=z_data,
                                        boundaries=boundaries,
                                        overlaps=overlaps,
                                        system="cylindrical")
                            )

    # Pre-compute and store full particle volumes
    full_particle_volumes = (4/3) * np.pi * (radii)**3

    # Get masks for particles completely inside or outside boundaries
    inside_mask = is_inside_boundaries(x_data=None, y_data=None,
                                       r_data=r_data, theta_data=theta_data,
                                       z_data=z_data,
                                       boundaries=boundaries,
                                       radii=radii,
                                       factor=factor,
                                       system="cylindrical")
    
    outside_mask = is_outside_boundaries(x_data=None, y_data=None,
                                         r_data=r_data, theta_data=theta_data,
                                         z_data=z_data,
                                         boundaries=boundaries,
                                         radii=radii,
                                         factor=factor,
                                         system="cylindrical")

    full_cylindrical_cell = boundaries["r_min"] < 0

    if full_cylindrical_cell and accurate_cylindrical is None:
        accurate_cylindrical = True

    # Initialise volume
    total_particle_volume = 0.0

    # Add volumes for fully inside particles
    total_particle_volume += np.sum(full_particle_volumes[inside_mask])

    # Skip fully outside particles (0 volume contribution)

    # Loop only over particles which are neither outside nor inside
    neither_mask = ~(inside_mask | outside_mask)

    if accurate_cylindrical:
        total_particle_volume += sum(
            calculate_particle_volume_cyl(i, radii, active_overlap_values,
                                          r_data)
                                     for i in np.where(neither_mask)[0]
                                    )
    else:
        total_particle_volume += sum(
            calculate_particle_volume(i, radii, active_overlap_values)
                                     for i in np.where(neither_mask)[0]
                                    )

    cell_volume = compute_cell_volume(boundaries=boundaries,
                                      system="cylindrical",
                                      cylinder_radius=None)

    packing_density = total_particle_volume/cell_volume

    return packing_density


def calculate_particle_volume_cyl(i, radii, active_overlap_values,
                                  r_data=None) -> float:
    """
    Calculate the volume contribution of a given particle.
    If the particle is fully within the boundaries, its entire volume is
    counted; likewise, if it is fully outside, no volume is counted.
    The main functionality lies between these two cases, where the volume of
    intersection between the particle and the boundary region is calculated.

    Args:
        i (int): Index of the particle being evaluated.
        total_particles (int): Total number of particles.
        radii (np.ndarray): Array of particle radii.
        inside_mask (np.ndarray): Boolean mask indicating particles completely
        inside the boundaries.
        outside_mask (np.ndarray): Boolean mask indicating particles completely
        outside the boundaries.
        full_particle_volumes (np.ndarray): Array of full particle volumes.
        active_overlap_values (np.ndarray): Array of overlap distances with
        boundaries.
        r_data (np.ndarray): Used to calculate sphere cylinder overlaps if
                             accurate_cylindrical is demanded.
        accurate_cylindrical (bool): Determines whether to accurately calculate
                                     sphere-cylinder overlaps or to approximate
                                     them as sphere-plane overlaps.

    Returns:
        float: The volume contribution of the particle.
    """

    # Initialise partial volume at zero
    partial_volume = 0.0

    # Extract overlaps in each dimension, taking the first non-NaN value
    r_overlap = next((val for val in active_overlap_values[i][0:2]
                        if not np.isnan(val)), None)
    theta_overlap = next((val for val in active_overlap_values[i][2:4]
                            if not np.isnan(val)), None)
    z_overlap = next((val for val in active_overlap_values[i][4:6]
                        if not np.isnan(val)), None)

    # Determine r overlap type (min or max) and its boundary
    if not np.isnan(active_overlap_values[i][0]):
        overlaps_r, min_boundary = True, True
    elif not np.isnan(active_overlap_values[i][1]):
        overlaps_r, min_boundary = True, False
    else:
        overlaps_r = False

    # Check overlaps in theta and z dimensions
    overlaps_theta = any(not np.isnan(val)
                            for val in active_overlap_values[i][2:4])

    overlaps_z = any(not np.isnan(val)
                        for val in active_overlap_values[i][4:6])

    # Calculate the total number of overlaps
    number_of_overlaps = overlaps_r + overlaps_theta + overlaps_z

    # Determine partial volume depending on the number of boundaries
    if number_of_overlaps == 1:
        if overlaps_r:
            # Cylinder sphere overlap
            partial_volume = sphere_cylinder_intersection(
                radii[i], r_data[i], r_overlap, min_boundary)
            # partial_volume = single_cap_intersection(
            #     radii[i], r_overlap)
        elif overlaps_z:
            partial_volume = single_cap_intersection(
                radii[i], z_overlap)
        else:
            partial_volume = single_cap_intersection(
                radii[i], theta_overlap)

    elif number_of_overlaps == 2:
        if overlaps_z and overlaps_theta:
            partial_volume = double_cap_intersection(
                radii[i], z_overlap, theta_overlap)
        elif overlaps_theta and overlaps_r:
            partial_volume = double_cap_intersection(
                radii[i], r_overlap, theta_overlap)
        elif overlaps_r and overlaps_z:
            # Cylinder-sphere-plane overlap
            partial_volume = sphere_cylinder_plane_intersection(
                radii[i], r_data[i], r_overlap, z_overlap, min_boundary
            )
        else:
            # Could potentially overlap radial boundary twice if the ring
            # is thin enough, have not got functionality for this yet
            # Also could potentially overlap theta boundary twice if they
            # are close enough together
            return 0

    elif number_of_overlaps == 3:
        if overlaps_r and overlaps_theta and overlaps_z:
            partial_volume = triple_cap_intersection(
                radii[i], r_overlap, theta_overlap, z_overlap)
        else:
            return 0

    # Return calculated particle volume
    return partial_volume


def generate_cylindrical_mesh(radius, base_level, height, r_divisions,
                              z_divisions, theta_divisions=None,
                              constant_volume=True):
    """
    Generate a cylindrical mesh with division indices, including a special
    inner cell at each z-layer.

    Args:
        radius (float): Radius of the cylinder.
        base_level (float): Base level of the cylinder (z_min).
        height (float): Height of the cylinder (z_max - z_min).
        r_divisions (int): Number of radial divisions.
        theta_divisions (int, optional): Number of angular divisions.
        z_divisions (int): Number of vertical divisions.
        constant_volume (bool, optional): Set cells to constant volume or not.
                                          Only compatible with theta_divisions
                                          = 3 (default)

    Returns:
        list: A list of tuples, each containing (indices, boundaries).
              Indices are (i, j, k), and boundaries are dictionaries.
    """

    # Calculate radius_inner for the central cylindrical cell
    if constant_volume:
        if theta_divisions is not None and theta_divisions != 3:
            raise ValueError("""theta_divisions must equal 3 for constant
                             volume cells and constant radial divisions.
                             (Dynamic radial divisions not yet supported)""")
        else:
            theta_divisions = 3
            radius_inner = radius / r_divisions
    else:
        radius_inner = radius / r_divisions

    # Radial boundaries
    radial_bounds = np.linspace(radius_inner, radius, r_divisions)

    # Vertical boundaries
    z_bounds = np.linspace(base_level, base_level + height, z_divisions + 1)

    # Target volume for cells in first radial layer
    radius_factor = radial_bounds[1]**2 - radius_inner**2
    cell_height = height / z_divisions
    target_volume = np.pi * radius_factor * cell_height / theta_divisions

    # Store all mesh boundaries
    mesh_boundaries = []

    # Loop through z-layers first
    for k in range(len(z_bounds) - 1):
        z_min, z_max = z_bounds[k], z_bounds[k + 1]

        # Add the special inner cell for this z-layer
        mesh_boundaries.append((
            (0, 0, k),  # Special inner cell indices for this layer
            # This configuration specifies a full cylindrical cell
            {
                "r_min": -radius_inner,
                "r_max": radius_inner,
                "theta_min": 0,
                "theta_max": 2 * np.pi,
                "z_min": z_min,
                "z_max": z_max,
            }
        ))

        # Loop through radial layers outside the special cell
        for i in range(len(radial_bounds) - 1):
            r_min, r_max = radial_bounds[i], radial_bounds[i + 1]
            layer_volume = (np.pi * (r_max**2 - r_min**2) * (z_max - z_min))

            # Calculate required angular divisions for this layer to maintain
            # constant volume
            if theta_divisions == 1:
                layer_theta_divisions = 1
            else:
                layer_theta_divisions = max(1, int(round(
                    layer_volume/target_volume)))

            # Angular boundaries for this layer
            theta_bounds = np.linspace(0, 2 * np.pi, layer_theta_divisions + 1)

            # Loop through angular divisions
            for j in range(len(theta_bounds) - 1):
                theta_min, theta_max = theta_bounds[j], theta_bounds[j + 1]

                # Append regular cell boundaries and indices
                mesh_boundaries.append((
                    # Adjust indices since the inner cell is (0, 0, k)
                    (i + 1, j, k),
                    {
                        "r_min": r_min,
                        "r_max": r_max,
                        "theta_min": theta_min,
                        "theta_max": theta_max,
                        "z_min": z_min,
                        "z_max": z_max,
                    }
                ))

    return mesh_boundaries