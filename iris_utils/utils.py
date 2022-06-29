from shapely.geometry import MultiPoint
import numpy as np
from iris.exceptions import MergeError
import iris


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
    # TODO could probably be more than 3d.
    if len(cube.shape) == 3:
        cube_spatial_dims = cube.shape[1:]
    elif len(cube.shape) < 3:
        cube_spatial_dims = cube.shape
    else:
        raise ValueError("Not able to handle cube dimension.")

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


def merge_aeq_cubes(cubes):
    """Merge almost equal cubes.
    Wrapper for CubesList.merge_cube() which first tries a normal merge_cube
    and if unsuccessful potentially check whether the coordinates are close.
    If the cubes are on different coordinate system a manual adjustment is needed.

    Arguments
    ---------
    cubes : CubeList
        List of cubes to merge.

    """

    # First we try and merge the cubes.
    try:
        return cubes.merge_cube()

    # If this doesn't work, we have to figure out what is wrong.
    except MergeError:
        # The first cube in the list is used a base.
        cube0 = cubes[0]
        cube0_lat_points = cube0.coord("grid_latitude").points
        cube0_lon_points = cube0.coord("grid_longitude").points

        # Loop over the rest of the cubes.
        for i, cube in enumerate(cubes[1:]):
            candidate_lat_points = cube.coord("grid_latitude").points
            # Check the coordinate system against the base cube.
            if cube.coord_system() != cube0.coord_system():
                print(
                    f"Coordinate system mismatch at cube {i+1}: "
                    " {cube.coord_system()} vs. {cube0.coord_system()}"
                )
            # If coord system match it is likely a "precision" error.
            # We check if the points array are not equal but they are close.
            elif (np.all(np.isclose(cube0_lat_points, candidate_lat_points))) and not (
                np.all(np.equal(cube0_lat_points, candidate_lat_points))
            ):
                print("Converting coordinates.")
                # Set the points of the candidate to the points of the base cube.
                cube.coord("grid_latitude").points = cube0_lat_points.copy()
                cube.coord("grid_longitude").points = cube0_lon_points.copy()
                # And bounds
                cube.coord("grid_latitude").bounds = cube0.coord(
                    "grid_latitude"
                ).bounds.copy()
                cube.coord("grid_longitude").bounds = cube0.coord(
                    "grid_longitude"
                ).bounds.copy()
                # Also have to overwrite the aux coord.
                cube.coord("latitude").points = cube0.coord("latitude").points.copy()
                cube.coord("longitude").points = cube0.coord("longitude").points.copy()
                # And bounds
                cube.coord("latitude").bounds = cube0.coord("latitude").bounds.copy()
                cube.coord("longitude").bounds = cube0.coord("longitude").bounds.copy()

            # If still not matching.
            if not cube.coord("grid_latitude") == cube0.coord("grid_latitude"):
                # Make sure long names match
                cube.coord("grid_latitude").long_name = cube0.coord(
                    "grid_latitude"
                ).long_name
                cube.coord("grid_longitude").long_name = cube0.coord(
                    "grid_longitude"
                ).long_name
            # Calendar might be different as well.
            if not cube0.coord("time").units == cube.coord("time").units:
                print("Converting calendar.")
                # Convert the calendar of the candidate cube to the base cube.
                cube.coord("time").convert_units(cube0.coord("time").units)

        # With this done we try to merge the list again.
        return cubes.merge_cube()


def attribute_to_aux(cubes, attribute_names=["driving_model_id", "model_id"]):
    """Add an attribute from the cube as a scalar coordinate.

    Arguments
    ---------
    cubes : iris.CubeList
        List of cubes to perform the operation on.
    attribute_names : list(string, string)
        List of names of the attributes to use for the new coordinate.
    """

    # Loop over all the cubes.
    for cube in cubes:
        # Create a new coordinate from the attribute.
        coord = (
            cube.attributes[attribute_names[0]]
            + "--"
            + cube.attributes[attribute_names[1]]
        )
        new_aux_coord = iris.coords.AuxCoord(coord, var_name="model_conf")
        # Add it to the cube.
        cube.add_aux_coord(new_aux_coord)
