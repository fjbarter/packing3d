# __init__.py

__version__ = "0.1.0"
__author__ = "Freddie Barter"
__email__ = "fjbarter@outlook.com"


"""
Packing Density Calculator Package
==================================
This package provides tools for calculating the packing density of particles
in 3D Cartesian and cylindrical coordinate systems.

Modules:
--------
- cartesian: Core Cartesian packing calculations.
- cylindrical: Core cylindrical packing calculations.
- io: Input/Output utilities for reading and processing data.
- geometry: Limited to `convert_to_cylindrical` for coordinate transformations.
"""

# Importing key functions for the public API

# Cartesian functions
from .cartesian import (
    compute_packing_cartesian,
    generate_cartesian_mesh,
)

# Cylindrical functions
from .cylindrical import (
    compute_packing_cylindrical,
    generate_cylindrical_mesh,
)

# Geometry function (limited public access)
from .geometry import convert_to_cylindrical

# IO functions
from .io import (
    read_vtk_file,
    retrieve_coordinates,
)

# Public API
__all__ = [
    "compute_packing_cartesian",
    "generate_cartesian_mesh",
    "compute_packing_cylindrical",
    "generate_cylindrical_mesh",
    "convert_to_cylindrical",
    "read_vtk_file",
    "retrieve_coordinates",
]