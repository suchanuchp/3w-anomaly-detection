"""Credits: https://github.com/petrobras/3W/blob/68ab92cb16198b1f10fb92dd5ba8fa9e188a835f/toolkit/base.py#L78"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import itertools

from preprocess.const import PATH_DATASET, PATH_DATASET_INI, COLUMNS_DATA_FILES


def label_and_file_generator_by_label(real=True, simulated=False, drawn=False):
    """returns mapping of label and list of file paths, Dict[int, List[Tuple[int, str]]"""
    ldict = {}
    for i in Path(PATH_DATASET).iterdir():
        try:
            # Considers only directories
            if i.is_dir():
                label = int(i.stem)
                for fp in i.iterdir():
                    path = str(fp)
                    # Considers only csv files
                    if fp.suffix == ".csv":
                        # Considers only instances from the requested
                        # source
                        if (
                                (simulated and fp.stem.startswith("SIMULATED"))
                                or (drawn and fp.stem.startswith("DRAWN"))
                                or (
                                real
                                and (not fp.stem.startswith("SIMULATED"))
                                and (not fp.stem.startswith("DRAWN"))
                        )
                        ):
                            if label in ldict.keys():
                                ldict[label] = ldict[label] + [(label, fp)]
                            else:
                                ldict[label] = [(label, fp)]


        except:
            # Otherwise (e.g. files or directory without instances), do
            # nothing
            pass
    return ldict


def label_and_file_generator_by_well(real=True, simulated=False, drawn=False):
    """returns mapping of well and list of file paths, Dict[int, List[Tuple[int, str]]"""
    wdict = {}
    for i in Path(PATH_DATASET).iterdir():
        try:
            # Considers only directories
            if i.is_dir():
                label = int(i.stem)
                for fp in i.iterdir():
                    path = str(fp)
                    well_name = str(path)[str(path).rfind('/')+1:str(path).rfind('_')]
                    # Considers only csv files
                    if fp.suffix == ".csv":
                        # Considers only instances from the requested
                        # source
                        if (
                                (simulated and fp.stem.startswith("SIMULATED"))
                                or (drawn and fp.stem.startswith("DRAWN"))
                                or (
                                real
                                and (not fp.stem.startswith("SIMULATED"))
                                and (not fp.stem.startswith("DRAWN"))
                        )
                        ):
                            if well_name in wdict.keys():
                                wdict[well_name] = wdict[well_name] + [(label, fp)]
                            else:
                                wdict[well_name] = [(label, fp)]


        except:
            # Otherwise (e.g. files or directory without instances), do
            # nothing
            pass
    return wdict


def get_all_labels_and_files(by_label=True):
    """Gets lists with tuples related to all real, simulated, or
    hand-drawn instances contained in the 3w dataset. Each list
    considers instances from a single source. Each tuple refers to a
    specific instance and contains its label (int) and its full path
    (Path).
    Returns:
        tuple: Tuple containing three lists with tuples related to real,
            simulated, and hand-drawn instances, respectively.
    """
    if by_label:
        real_instances = label_and_file_generator_by_label(real=True, simulated=False, drawn=False)
        simulated_instances = label_and_file_generator_by_label(real=False, simulated=True, drawn=False)
        drawn_instances = label_and_file_generator_by_label(real=False, simulated=False, drawn=True)
    else:
        real_instances = label_and_file_generator_by_well(real=True, simulated=False, drawn=False)
        simulated_instances = label_and_file_generator_by_well(real=False, simulated=True, drawn=False)
        drawn_instances = label_and_file_generator_by_well(real=False, simulated=False, drawn=True)

    return real_instances, simulated_instances, drawn_instances


def load_instance(instance):
    """Loads all data and metadata from a specific `instance`.
    Args:
        instance (tuple): This tuple must refer to a specific `instance`
            and contain its label (int) and its full path (Path).
    Raises:
        Exception: Error if the CSV file passed as arg cannot be read.
    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the CSV file. Its columns contain data loaded from the other
            columns of the CSV file and metadata loaded from the
            argument `instance` (label, well, and id).
    """
    # Loads label metadata from the argument `instance`
    label, fp = instance

    try:
        # Loads well and id metadata from the argument `instance`
        well, id = fp.stem.split("_")

        # Loads data from the CSV file
        df = pd.read_csv(fp, index_col="timestamp", parse_dates=["timestamp"])
        assert (
                df.columns == COLUMNS_DATA_FILES[1:]
        ).all(), f"invalid columns in the file {fp}: {df.columns.tolist()}"
    except Exception as e:
        raise Exception(f"error reading file {fp}: {e}")

    # Incorporates the loaded metadata
    df["label"] = label
    df["well"] = well
    df["id"] = id

    # Incorporates the loaded data and ordenates the df's columns
    df = df[["label", "well", "id"] + COLUMNS_DATA_FILES[1:]]

    return df


def load_data_dict(instance_dict, n_instances_per_label=None):
    """Loads all data and metadata from a specific `instance`.
    Args:
        instances (list of tuples): list of `instance` tuple, containing its label (int) and its full path (Path).
        n (Optional[int]): number of instances to load
    Raises:
        Exception: Error if the CSV file passed as arg cannot be read.
    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the CSV file. Its columns contain data loaded from the other
            columns of the CSV file and metadata loaded from the
            argument `instance` (label, well, and id).
    """
    # well_names = set([str(path)[str(path).rfind('/')+1:str(path).rfind('_')] for _, path in instances])
    # well_names = list(itertools.islice(well_names, n_well)) if n_well else well_names
    idict = dict()
    # n = n_well if n_well else len(instances)

    for label, instances in tqdm(list(instance_dict.items())):
        for instance in instances[:n_instances_per_label]:
            _, path = instance
            df = load_instance(instance)
            df['start'] = 0
            df.loc[df.iloc[0].name, 'start'] = 1
            if label in idict.keys():
                existing_df = idict[label]
                idict[label] = pd.concat([existing_df, df])
            else:
                idict[label] = df
    return idict


def load_well_dict(instances, n_well=None, max_instances_per_well=None):
    # well_names = set([str(path)[str(path).rfind('/')+1:str(path).rfind('_')] for _, path in instances])
    # well_names = list(itertools.islice(well_names, n_well)) if n_well else well_names
    idict = dict()
    n_well = n_well if n_well else len(instances)

    for well_name, instances in tqdm(list(instances.items())[:n_well]):
        max_instances_per_well = max_instances_per_well if max_instances_per_well else len(instances)
        for instance in instances[:max_instances_per_well]:
            _, path = instance
            df = load_instance(instance)
            if well_name in idict.keys():
                existing_df = idict[well_name]
                idict[well_name] = pd.concat([existing_df, df])
            else:
                idict[well_name] = df
    return idict


def load_instance_dict(instances, n=None):
    """Loads all data and metadata from a specific `instance`.
    Args:
        instances (list of tuples): list of `instance` tuple, containing its label (int) and its full path (Path).
        n (Optional[int]): number of instances to load
    Raises:
        Exception: Error if the CSV file passed as arg cannot be read.
    Returns:
        pandas.DataFrame: Its index contains the timestamps loaded from
            the CSV file. Its columns contain data loaded from the other
            columns of the CSV file and metadata loaded from the
            argument `instance` (label, well, and id).
    """

    idict = dict()
    n = n if n else len(instances)
    for instance in tqdm(instances[:n]):
        df = load_instance(instance)
        well_name = df.iloc[0].well
        if well_name in idict.keys():
            existing_df = idict[well_name]
            idict[well_name] = pd.concat([existing_df, df])
        else:
            idict[well_name] = df
    return idict
