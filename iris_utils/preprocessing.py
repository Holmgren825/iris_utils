import csv

import iris
import iris.pandas
import pandas as pd
import regex as re
from tqdm.autonotebook import tqdm

# TODO This will get more conversions as we use more variables.
PARAM_CONV = {
    "Lufttemperatur": {
        "standard_name": "air_temperature",
        "Maximum": "tasmax",
        "Minimum": "tasmin",
        "Average": "tas",
    },
}


def load_df(file_path: str) -> pd.DataFrame:
    """Load a pandas dataframe with station data from Mora. Will parse and read attributes such as station name
    variable name, standard name.

    Arguments
    ---------
    file_path: str
        Path to file (csv).

    Returns
    -------
    df: pandas.Dataframe

    """

    # We need to learn some things about the csv before loading with pandas since header length will vary.
    with open(file_path) as file:
        reader = csv.reader(file)
        prev_line = ""
        found_attrs = {}
        for i in range(20):
            line = next(reader)[0]
            # Find station name
            if re.match(r"Station name[^,]+", prev_line):
                found_attrs["station_name"] = line.split(";")[0]
            if group := re.match(r"Parameter;\"([^,]+)\"", line):
                found_attrs["parameter"] = group.groups()[0]
            if group := re.match(r"Unit, database;\"([^,]+)\"", line):
                found_attrs["unit"] = group.groups()[0]
            if group := re.match(r"Statistics type;\"([^,]+)\"", line):
                found_attrs["var_name"] = group.groups()[0]

            prev_line = line
            # Find header line.
            if re.match(r"Time;Offset[^,]+", line):
                header = i
                break
    # Now use pandas to read the csv
    df = pd.read_csv(file_path, header=header, delimiter=";")
    df.index = pd.to_datetime(df["Time"])
    df = df.rename_axis("time")
    df["Represents"] = pd.to_datetime(df["Represents"])
    # Keep the station name
    df.attrs = found_attrs

    return df


def cube_from_mora_csv(file_path: str) -> iris.cube.Cube:
    """Prepare an iris cube with station data from the Mora database (csv file).

    Arguments
    ---------
    file_path: str
        Path to file.

    Returns
    -------
    cube
    """
    df = load_df(file_path)

    # TODO This does not work for all stations for some unknown reason.
    # cube = iris.pandas.as_cubes(
    #     df, aux_coord_cols=["Represents"], ancillary_variable_cols=["Quality"]
    # )
    cube = iris.pandas.as_cubes(df)
    cube = cube[3]

    cube.standard_name = PARAM_CONV[df.attrs["parameter"]]["standard_name"]
    cube.var_name = PARAM_CONV[df.attrs["parameter"]][df.attrs["var_name"]]
    cube.units = df.attrs["unit"]
    cube.attributes["station_name"] = df.attrs["station_name"]
    cube.data = cube.lazy_data()

    return cube


def cubes_from_mora_csv(file_paths: list) -> iris.cube.CubeList:
    """Read multiple More csv files into a iris.cube.CubeList

    Arguments
    ---------
    file_paths: list
        List of file paths (str)

    Returns
    -------
    cubes: iris.cube.CubeList
    """

    cubes = iris.cube.CubeList()

    for file_path in tqdm(file_paths):
        cubes.append(cube_from_mora_csv(file_path))

    return cubes
