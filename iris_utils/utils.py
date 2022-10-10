import copy
from multiprocessing import Pool

import cartopy.crs as ccrs
import dask.array as da
import iris
import iris.analysis.cartography
import numpy as np
from iris.exceptions import MergeError
from shapely.geometry import MultiPoint


def get_weights(cube):
    """Get area weights for a cube, even if auxilary coordinaties are present.

    Arguments
    ---------
    cube : iris.cube.Cube

    Returns
    -------
    weights : array_like
    """

    #  First we try and get the weights
    try:
        weights = iris.analysis.cartography.area_weights(cube)
    # If we fail it is likely due to multiple coordinates for lats and longs.
    except ValueError as e:
        # If we get the right message.
        if (
            str(e)
            == "Calling `_get_lon_lat_coords` with multiple lat or lon coords is currently disallowed"
        ):
            # Copy the coords
            lats = cube.coord("latitude").copy()
            longs = cube.coord("longitude").copy()
            # Then we remove them.
            cube.remove_coord("latitude")
            cube.remove_coord("longitude")
            # Recurse to get the weights.
            return get_weights(cube)
        elif (
            str(e)
            == "Coordinates 'grid_latitude' and 'grid_longitude' must have bounds to determine the area weights."
        ):
            cube.coord("grid_latitude").guess_bounds()
            cube.coord("grid_longitude").guess_bounds()
            # Recursion
            return get_weights(cube)

    return weights


def mask_from_shape(
    cube, shape, coord_system=True, coord_names=("latitude", "longitude")
):
    """Create a iris cube compatible mask from a polygon.

    Arguemnts
    ---------
    cube : iris.cube.Cube
        Iris cube to create the mask for.
    shape : shapely.geometry.polygon.Polygon.
        Shapefile outlining the region to use as a mask.
    coord_system : bool, default: True
        Does the cube have a coordinate system?
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
    if coord_system:
        # It is likely that the shape and the cube don't share coordinate system.
        # Hence we should make sure and convert coords of cube before selecting.
        # We assume that shape is in PlateCarree
        shape_projection = ccrs.PlateCarree()
        # Get the projection of the cube as a cartopy crs.
        cube_projection = cube.coord_system().as_cartopy_projection()
        # Transform the cube grid to the shape projection.
        transformed_points = shape_projection.transform_points(
            cube_projection, x.flatten(), y.flatten()
        )
        # Extract the points
        x_flat = transformed_points[:, 0]
        y_flat = transformed_points[:, 1]
    else:
        x_flat = x.flatten()
        y_flat = y.flatten()

    # Create shapely points
    lon_lat_points = np.vstack([x_flat, y_flat])
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


def mask_cube(cube, mask):
    """Mask data in a cube with a boolean mask. Unlike the built in iris util,
    this keeps the data lazy. Modifies the cube inplace.

    Arguments
    --------
    cube : iris.cube.Cube
        n dimensional instance of an iris cube.
    mask : numpy.ndarray of bool
        A n dimensional boolean array of Truth values representing points to mask
    """

    # Simply overwrite the cube data with a masked dask array.
    # NOTE: If we don't want to do this inplace, what to we do? Copy the original cube?
    cube.data = da.ma.masked_array(cube.core_data(), mask)


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
                cube.coord("grid_latitude").points = copy.deepcopy(cube0_lat_points)
                cube.coord("grid_longitude").points = copy.deepcopy(cube0_lon_points)
                # And bounds
                cube.coord("grid_latitude").bounds = copy.deepcopy(
                    cube0.coord("grid_latitude").bounds
                )
                cube.coord("grid_longitude").bounds = copy.deepcopy(
                    cube0.coord("grid_longitude").bounds
                )
                # Also have to overwrite the aux coord.
                cube.coord("latitude").points = copy.deepcopy(
                    cube0.coord("latitude").points
                )
                cube.coord("longitude").points = copy.deepcopy(
                    cube0.coord("longitude").points
                )
                # And bounds
                cube.coord("latitude").bounds = copy.deepcopy(
                    cube0.coord("latitude").bounds
                )
                cube.coord("longitude").bounds = copy.deepcopy(
                    cube0.coord("longitude").bounds
                )

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


def attribute_to_aux(
    cubes,
    attribute_names=[
        "driving_model_id",
        "model_id",
        "driving_model_ensemble_member",
    ],
    missing_keys_ind=[0, 2],
    new_coord_name="ens_id",
):
    """Add any number of attributes from the cube as a scalar coordinate.
    Useful for merging cubes.

    Arguments
    ---------
    cubes : iris.CubeList
        List of cubes to perform the operation on.
    attribute_names : list
        List of keys to the cube attributes used for the new categorical coordinate.
        Default: driving_model_id, model_id, driving_model_ensemble_member.
    missing_key_ind : list
        Which indices of attribute_names to use if a cube is missing a key.
        Generally requires some investigation of the cubes.
        Default: 0, 2
    new_coord_name : string
        Name of the new coordinate.
    """

    # Loop over all the cubes.
    for i, cube in enumerate(cubes):
        # Create a new coordinate value from the attributes.
        try:
            # Get all the attributes.
            new_coord_data = [cube.attributes[key] for key in attribute_names]
            # Join them to one string, separated by --.
            new_coord_data = "--".join(new_coord_data)
        # If the key doesn't exist
        except KeyError:
            print(f"Cube {i} missing a key, skipping key.")
            # Here we instead only select attribute names that should be available.
            new_coord_data = [
                cube.attributes[attribute_names[ind]] for ind in missing_keys_ind
            ]
            # Again, join.
            new_coord_data = "--".join(new_coord_data)

        finally:
            # Create a new AuxCoord.
            new_aux_coord = iris.coords.AuxCoord(
                new_coord_data, var_name=new_coord_name, long_name=new_coord_name
            )
            # Add it to the cube.
            cube.add_aux_coord(new_aux_coord)


def make_lazy(cube, inplace=True):
    """Make cube lazy again. This is useful since climix only accepts cubes with lazy data.

    Arguments:
    ----------
    cube : iris.cube.Cube
        The cube to make lazy.
    inplace : bool, default: True
        Modify the cube inplace.
    """
    if inplace:
        data = cube.data
        cube.data = da.asarray(data)
        # This should work.
        return
    else:
        new_cube = cube.copy()
        data = new_cube.data
        new_cube.data = da.asarray(data)

        return new_cube
