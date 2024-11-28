# This module contains the two functions required to read vtk files and
# retrieve coordinate data from them

from pyvista import read
import os

def read_vtk_file(file):  # Returns data
    """
    Returns data object after reading a file with PyVista. Includes a check
    to ensure the file exists.

    Args:
        file (str): Path to the .vtk file.

    Returns:
        pyvista.DataSet: The data object from the .vtk file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be read as a PyVista dataset.
    """
    # Check if file exists
    if not os.path.isfile(file):
        raise FileNotFoundError(
            f"The file '{file}' does not exist. Please provide a valid path.")

    # Attempt to read the file
    try:
        data = read(file)
    except Exception as e:
        raise ValueError(f"Error reading the file '{file}': {e}")

    return data


def retrieve_coordinates(data):  # Returns x_data, y_data, z_data, radii
    """
    Extract x, y, z coordinates and radii from the dataset.

    Args:
        data: PyVista data object containing particle information.

    Returns:
        tuple: Arrays for x, y, z coordinates and radii.
    """
    x_data = data.points[:, 0]
    y_data = data.points[:, 1]
    z_data = data.points[:, 2]
    radii = data["radius"]
    return x_data, y_data, z_data, radii

