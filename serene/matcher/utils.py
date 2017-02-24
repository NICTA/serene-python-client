import logging
from datetime import datetime

from serene.api.exceptions import InternalError


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
