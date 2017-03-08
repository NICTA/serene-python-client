"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the Octopus object
"""
import logging

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class OctopusInternal(object):
    """
        Octopus is the central integration map for a collection of DataSets.
    """
    def __init__(self, session):
        """
        Builds up an Octopus
        """
        self._session = session
