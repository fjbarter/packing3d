# Packing3D

A Python package for calculating the local and bulk packing density of particles in 3D Cartesian and cylindrical coordinate systems, with capability to mesh a region and visualise packing density distribution.

![Example Image1](https://github.com/fjbarter/packing3d/blob/main/source/Before_After_Vibration.png?raw=true)
Above image: Cross-sectional packing density distribution of 100,000 particles (~500-600 microns) in a cylindrical container of diameter 75 mm, with packing density calculated from z = [0.005 m, 0.020 m].

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Key Functions](#key-functions)
  - [compute_packing_cartesian](#compute_packing_cartesian)
  - [generate_cartesian_mesh](#generate_cartesian_mesh)
  - [compute_packing_cylindrical](#compute_packing_cylindrical)
  - [generate_cylindrical_mesh](#generate_cylindrical_mesh)
  - [convert_to_cylindrical](#convert_to_cylindrical)
  - [read_vtk_file](#read_vtk_file)
  - [retrieve_coordinates](#retrieve_coordinates)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Examples](#examples)
- [Limitations](#limitations)
- [Planned Features](#planned-features)
- [License](#license)
- [Contact](#contact)

---

## Requirements

- **Python 3.8 or later***
- ***WARNING: As of 28th Nov 2024, the VTK dependency of PyVista does not work with Python 3.13**

### Dependencies

- **NumPy**: For efficient numerical operations.
- **SciPy**: For numerical integration and advanced mathematical functions.
- **PyVista**: For reading and handling `.vtk` files.

---

## Installation

The package is available on PyPI and can be installed via:

```bash
pip install packing3d
```

Alternatively, clone the repository at https://github.com/fjbarter/packing3d/ and install manually from the local files.

---

## Key Functions

### `compute_packing_cartesian`

#### Description

Computes the packing density of particles within a defined cuboidal region in Cartesian coordinates.

#### Args

- `file` (*str*, optional): Path to the `.vtk` file containing particle data. Required if coordinate data is not provided.
- `boundaries` (*dict*, optional): Dictionary defining the cuboidal region boundaries with keys `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max`. Defaults to automatic boundaries based on the dataset.
- `x_data`, `y_data`, `z_data` (*np.ndarray*, optional): Preloaded x, y, z coordinates of particles.
- `radii` (*np.ndarray*, optional): Preloaded radii of particles.

#### Returns

- *float*: The packing density as the fraction of volume occupied by particles.

#### Example

```python
from packing3d import compute_packing_cartesian

packing_density = compute_packing_cartesian(file='particles.vtk')
print(f"Packing Density: {packing_density}")
```

---

### `generate_cartesian_mesh`

#### Description

Generates a Cartesian mesh that can approximate a cuboidal or cylindrical region. If cylindrical_mesh_shape is specified True, the exact volume of the cartesian cells which overlap the cylinder walls is calculated and used for the packing density, so there is no need to worry about incorrectly low packing densities at the wall (although they will still be low!).

#### Args

- `x_divisions`, `y_divisions`, `z_divisions` (*int*): Number of divisions along the x, y, and z axes.
- `boundaries` (*dict*, optional): Dictionary defining the boundaries of the Cartesian region.
- `cylindrical_mesh_shape` (*bool*, optional): Boolean that creates a cylindrical mesh shape with cartesian cells.
- `radius` (*float*, optional): Radius of the cylindrical region to approximate, if applicable.
- `base_level` (*float*, optional): Base level of the cylinder in the z-direction.
- `height` (*float*, optional): Height of the cylindrical region for meshing

#### Returns

- *list*: A list of tuples, each containing `(indices, boundaries)` for each cell.

#### Example

```python
from packing3d import generate_cartesian_mesh

mesh = generate_cartesian_mesh(x_divisions=10, y_divisions=10, z_divisions=5,
                               boundaries=boundaries, cylindrical_mesh_shape=False,
                               radius=None, base_level=None, height=None)
print(mesh)
```

---

### `compute_packing_cylindrical`

#### Description

Computes the packing density of particles within a defined cylindrical region in cylindrical coordinates.

#### Args

- `file` (*str*, optional): Path to the `.vtk` file containing particle data. Required if coordinate data is not provided.
- `boundaries` (*dict*, optional): Dictionary defining the cylindrical region boundaries with keys `r_min`, `r_max`, `theta_min`, `theta_max`, `z_min`, and `z_max`. Defaults to automatic boundaries based on the dataset.
- `r_data`, `theta_data`, `z_data` (*np.ndarray*, optional): Preloaded radial, angular, and z coordinates of particles.
- `radii` (*np.ndarray*, optional): Preloaded radii of particles.

#### Returns

- *float*: The packing density as the fraction of the cylindrical volume occupied by particles.

#### Example

```python
from packing3d import compute_packing_cylindrical

packing_density = compute_packing_cylindrical(file='particles.vtk')
print(f"Packing Density: {packing_density}")
```

---

### `generate_cylindrical_mesh`

#### Description

Generates a cylindrical mesh with division indices for radial, angular, and z-direction partitions. There is an inner cylindrical cell, created to avoid problems associated with converging radial lines.

The mesh cells created all have a constant volume, with theta_divisions determining the number of cells in the first ring, outside the inner cylindrical cell. If radius_inner is None, it will have a radius equal to the width of each radial division.

#### Args

- `radius` (*float*): Radius of the cylindrical region.
- `height` (*float*): Height of the cylindrical region.
- `r_divisions`, `theta_divisions`, `z_divisions` (*int*): Number of divisions in radial, angular, and z directions.
- 'radius_inner' (float, optional): Radius of the inner cylindrical cell. Defaults to radius / r_divisions.

#### Returns

- *list*: A list of tuples, each containing `(indices, boundaries)` for each cell.

#### Example

```python
from packing3d import generate_cylindrical_mesh

mesh = generate_cylindrical_mesh(radius=10, height=20, r_divisions=5, theta_divisions=8, z_divisions=10)
print(mesh)
```

---

### `convert_to_cylindrical`

#### Description

Converts Cartesian coordinates to cylindrical coordinates with theta in `[0, 2π]`.

#### Args

- `x_data` (*np.ndarray*): Array of x-coordinates.
- `y_data` (*np.ndarray*): Array of y-coordinates.

#### Returns

- `r_data` (*np.ndarray*): Radial distances from the origin.
- `theta_data` (*np.ndarray*): Angles in radians from the x-axis, in the range `[0, 2π]`.

---

### `read_vtk_file`

#### Description

Reads a `.vtk` file using PyVista and returns the data object.

#### Args

- `file` (*str*): Path to the `.vtk` file.

#### Returns

- *pyvista.DataSet*: The data object from the `.vtk` file.

---

### `retrieve_coordinates`

#### Description

Extracts x, y, z coordinates and radii from the dataset.

#### Args

- `data`: PyVista data object containing particle information.

#### Returns

- `x_data`, `y_data`, `z_data`, `radii` (*np.ndarray*): Arrays of coordinates and radii.

---

## How It Works

1. **Data Reading**:
   - Uses PyVista to read `.vtk` files containing particle coordinates and radii.
   - `read_vtk_file` and `retrieve_coordinates` facilitate this process.

2. **Coordinate Conversion**:
   - For cylindrical computations, Cartesian coordinates are converted using `convert_to_cylindrical`.

3. **Boundary Determination**:
   - Boundaries can be user-defined or automatically computed based on the dataset using `compute_automatic_boundaries`.

4. **Volume Calculations**:
   - For particles completely inside the boundaries, their full volume is added.
   - For particles partially overlapping boundaries, geometric functions compute the exact volume of intersection.

5. **Packing Density Computation**:
   - Total particle volume within the boundaries is divided by the cell volume to obtain the packing density.

---

## Testing

- **Unit Tests**: Planned for future releases to ensure accuracy and reliability.
- **Sample Data**: `.vtk` files are provided for testing in the examples directory.

---

## Examples

- `examples.py` in the examples directory provides three examples with corresponding vtk files.

---

## Limitations

- Optimised for spherical particles only.
- Requires `.vtk` files with properly formatted particle data.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, support, or contributions, reach out to:

- **Name**: Freddie Barter
- **Email**: [fjbarter@outlook.com](mailto:fjbarter@outlook.com)
