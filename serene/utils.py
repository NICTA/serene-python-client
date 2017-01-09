from .exceptions import InternalError
import logging
from datetime import datetime
import pandas as pd


def convert_datetime(datetime_string, fmt="%Y-%m-%dT%H:%M:%SZ"):
    """
    Convert string to datetime object.

    Args:
        datetime_string: string which contains datetime object;
        fmt: format which specifies the pattern of datetime format in the string.

    Returns: datetime on successful conversion.
    Raises: InternalDIError if conversion fails.

    """
    # here we put "%Y-%m-%dT%H:%M:%SZ" to be the default format
    # T and Z are fixed literals
    # T is time separator
    # Z is literal for UTC time zone, Zulu time
    try:
        converted = datetime.strptime(datetime_string, fmt)
    except Exception as e:
        logging.error("Failed converting string to datetime: " + repr(datetime_string))
        raise InternalError("Failed converting string to datetime: " + repr(datetime_string), e)
    return converted


def construct_labelData(matcher_dataset, filepath, header_column="column_name", header_label="class"):
    """
    We want to construct a dictionary {column_id:class_label} for the dataset based on a .csv file.
    This method reads in .csv as a Pandas data frame, selects columns "column_name" and "class",
    drops NaN and coverts these two columns into dictionary.
    We obtain a lookup dictionary
    where the key is the column name and the value is the class label.
    Then by using column_map the method builds the required dictionary.

    Args:
        filepath: string where .csv file is located.
        header_column: header for the column with column names
        header_label: header for the column with labels

    Returns: dictionary

    """
    label_data = {}  # we need this dictionary (column_id, class_label)
    try:
        frame = pd.read_csv(filepath, na_values=[""])
        # dictionary (column_name, class_label)
        name_labels = frame[[header_column, header_label]].dropna().set_index(header_column)[header_label].to_dict()
        column_map = [(col.id, col.name) for col in matcher_dataset.columns]
        for col_id, col_name in column_map:
            if col_name in name_labels:
                label_data[int(col_id)] = name_labels[col_name]
    except Exception as e:
        raise InternalError("construct_labelData", e)

    return label_data
