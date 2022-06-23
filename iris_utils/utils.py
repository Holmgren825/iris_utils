from shapely.geometry import MultiPoint
import numpy as np


def mask_from_shape(cube, shape, coord_names=("latitude", "longitude")):
    """Create a iris cube compatible mask from a polygon.

    Arguemnts
    ---------
    cube : iris.cube.Cube
        Iris cube to create the mask for.
    shape : shapely.geometry.polygon.Polygon.
        Shapefile outlining the region to use as a mask.
    coord_names : tuple(string, string)
        Names of latitude and longitude in cube. Default to "latitude" and "longitude".
    Returns
    -------
    mask : numpy.ndarray
    """

    # Does the cube contain a 3rd dimesion?
    # TODO could probably be more than 3 dim.
    if len(cube.shape) == 3:
        cube_spatial_dims = cube.shape[1:]
    elif len(cube.shape) < 3:
        cube_spatial_dims = cube.shape
    else:
        raise ValueError("Cube dimension not know.")

    # Create a meshgrid from the cube coords.
    x, y = np.meshgrid(
        cube.coord(coord_names[1]).points, cube.coord(coord_names[0]).points
    )

    # Create shapely points
    lon_lat_points = np.vstack([x.flat, y.flat])
    points = MultiPoint(lon_lat_points.T)

    # Check which indices are within the shapefile
    indices = [i for i, p in enumerate(points.geoms) if shape.contains(p)]

    # Create the mask
    mask = np.ones(cube_spatial_dims, dtype=bool)
    # Set values within the specified region to false, e.g. the areas we want to keep.
    mask[np.unravel_index(indices, mask.shape)] = False

    # If cube is three dimensional e.g. incuding time.
    # we broadcast the mask along time.
    if len(cube.shape) == 3:
        mask = np.broadcast_to(mask, cube.shape)

    return mask
