"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the Octopus object
"""
import logging

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class Octopus(object):
    """
        Octopus is the central integration map for a collection of DataSets.
    """
    def __init__(self,
                 ssds,
                 name="",
                 description="",
                 feature_config=None,
                 model_type="randomForest",
                 resampling_strategy="NoResampling",
                 num_bags=10,
                 bag_size=10,
                 ontologies=None,
                 modeling_props=None):
        """

        :param ssds:
        :param name:
        :param description:
        :param feature_config:
        :param model_type:
        :param resampling_strategy:
        :param num_bags:
        :param bag_size:
        :param ontologies:
        :param modeling_props:
        """
        self._id = None
        self._stored = False
        self._date_created = None
        self._date_modified = None

        self._ssds = ssds
        self._ontologies = ontologies
        self._name = name
        self._description = description
        self._feature_config = feature_config
        self._model_type = model_type
        self._resampling_strategy = resampling_strategy
        self._num_bags = num_bags
        self._bag_size = bag_size
        self._modeling_props = modeling_props

    def from_json(self, data, ssd_endpoint, ontology_endpoint):
        pass