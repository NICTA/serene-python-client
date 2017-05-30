import logging
import random
import string
from datetime import datetime
import json

from serene.api.exceptions import InternalError

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


def gen_id(k=4):
    """
    Generates a random temporary id. Note that this is alpha-numeric.
    Alpha character ids are used to indicate that this ontology is not
    stored.

    :return:
    """
    return ''.join(random.sample(string.ascii_lowercase, k))


def flatten(xs):
    """Flatten a 2D list"""
    return [x for y in xs for x in y]


class Searchable(object):
    """
    A small object that allows hierarchical search across its properties...
    """
    getters = []

    @classmethod
    def search(cls, container, item, exact=False):
        """
        Finds an item in a container. The search method enables
        shorthand searches based on uniqueness. e.g. if there is only one
        DataNode with name='id' then it can be referenced with DataNode('id').
        Though if there are two DataNode('id', ClassNode('Person')) and
        DataNode('id', ClassNode('Business')), then the ClassNode would need
        to be provided to remove the ambiguity. Search will use the 'getter'
        objects provided to compare keys.

        The search method is called from a class type on a container and
        target.

        ClassNode.search(class_node_list, class_node)

        The target can be partially complete e.g. Class("Person")

        ClassNode.search(class_node_list, Class("Person")

        :param container: An iterable of objects
        :param item: The target object to find, can be partially complete.
        :return: the item or None or LookupError if ambiguous
        """

        def find(haystack, needle, getter):
            """
            This function will look into the 'haystack' container for the item
            'needle' using the getter function

            Note: None is treated as a wildcard e.g. DataNode(None, 'name') == DataNode('Person', 'name')

            :param haystack:
            :param needle:
            :param getter:
            :return:
            """
            matches = []
            for dn in haystack:
                try:
                    if getter(needle) is None or getter(dn) is None:
                        matches.append(dn)
                    elif getter(needle) == getter(dn):
                        matches.append(dn)
                except AttributeError:
                    # if the getter fails, then we add nothing...
                    pass

            if len(matches) == 0:
                # not found...
                return False, None
            elif len(matches) == 1:
                # item found!
                return True, matches[0]
            else:
                # found ambiguities...
                return False, matches

        # now we iterate over all the getters until we find our match...
        candidates = container
        for get in cls.getters:
            ok, match = find(candidates, item, get)
            if ok:
                if exact and match == item:
                    return match
                elif exact and match != item:
                    return None
                else:
                    return match
            if match is None:
                return None
            else:
                # a match is found, but it is ambiguous
                candidates = match

        msg = "Failed to find item. {} is ambiguous: {}".format(item, candidates)
        _logger.error(msg)
        raise LookupError(msg)


def convert_datetime(datetime_string, fmt="%Y-%m-%dT%H:%M:%S.%f"):
    """
    Convert string to datetime object.

    Args:
        datetime_string: string which contains datetime object;
        fmt: format which specifies the pattern of datetime format in the string.

    Returns: datetime on successful conversion.
    Raises: InternalError if conversion fails.

    """
    # here we put "%Y-%m-%dT%H:%M:%S.%f" to be the default format
    # T and Z are fixed literals
    # T is time separator
    # Z is literal for UTC time zone, Zulu time
    try:
        converted = datetime.strptime(datetime_string, fmt)
    except Exception as e:
        logging.error("Failed converting string to datetime: " + repr(datetime_string))
        raise InternalError("Failed converting string to datetime: " + repr(datetime_string), e)
    return converted


def get_prefix(node):
    """
    Strips off the name in the URI to give the prefixlabel...
    :param node: The full URI string
    :return: (prefix, label) as (string, string)
    """
    if '#' in node:
        name = node.split("#")[-1]
    else:
        # there must be no # in the prefix e.g. schema.org/
        name = node.split("/")[-1]
    return node[:-len(name)]


def get_label(node):
    """
    Strips off the prefix in the URI to give the label...
    :param node: The full URI string
    :return: (prefix, label) as (string, string)
    """
    if '#' in node:
        name = node.split("#")[-1]
    else:
        # there must be no # in the prefix e.g. schema.org/
        name = node.split("/")[-1]
    return name