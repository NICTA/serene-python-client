"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the Octopus object
"""
import logging
import time

from .semantics.ssd import SSD
from .semantics.ontology import Ontology
from .dataset import DataSet
from ..matcher import Status
from ..utils import convert_datetime
from ..matcher import ModelState

_logger = logging.getLogger()
_logger.setLevel(logging.WARN)


class OctopusScore(object):
    """
    The score object from the prediction
    """
    def __init__(self, json):
        """Converts the json blob to the score values"""
        # karma score values
        self.sizeReduction = json['sizeReduction']
        self.nodeConfidence = json['nodeConfidence']
        self.nodeCoherence = json['nodeCoherence']
        self.linkCoherence = json['linkCoherence']
        self.linkCost = json['linkCost']

        # weighted average of sizeReduction, nodeConfidence, nodeCoherence
        self.karmaScore = json['karmaScore']

        # order using all karma score values...
        self.karmaRank = json['karmaRank']

        # additional scores: percentage of original columns included in semantic model
        self.nodeCoverage = json['nodeCoverage']

    def __repr__(self):
        """Output string"""
        base = "Score(rank={:d}, score={:.2f}, confidence={:.2f}, coverage={:.2f})"
        return base.format(
            self.karmaRank,
            self.karmaScore,
            self.nodeConfidence,
            self.nodeCoverage)


class SSDResult(object):
    """Octopus Prediction result object"""
    def __init__(self, ssd, score):
        self._score = score
        self._ssd = ssd

    @property
    def score(self):
        return self._score

    @property
    def ssd(self):
        return self._ssd

    def __repr__(self):
        return "SSDResult({})".format(self.score.karmaRank)


class Octopus(object):
    """
        Octopus is the central integration map for a collection of DataSets.
    """
    def __init__(self,
                 ssds=None,
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

        self._model_id = None
        self._matcher = None
        self._modeling_props = None
        self._semantic_type_map = None
        self._state = None

        self._session = None
        self._dataset_endpoint = None
        self._model_endpoint = None
        self._ontology_endpoint = None
        self._ssd_endpoint = None

    def update(self, json, session, dataset_endpoint, model_endpoint, ontology_endpoint, ssd_endpoint):
        """Update the object using json..."""
        self._session = session
        self._dataset_endpoint = dataset_endpoint
        self._model_endpoint = model_endpoint
        self._ontology_endpoint = ontology_endpoint
        self._ssd_endpoint = ssd_endpoint

        # add the storage parameters...
        self._stored = True
        self._name = json['name']
        self._description = json['description']
        self._date_created = convert_datetime(json['dateCreated'])
        self._date_modified = convert_datetime(json['dateModified'])
        self._id = json['id']

        # build out the octopus parameters
        self._ssds = [self._ssd_endpoint.get(o) for o in json['ssds']]
        self._ontologies = [self._ontology_endpoint.get(o) for o in json['ontologies']]
        self._model_id = json['lobsterID']
        self._matcher = self._model_endpoint.get(self._model_id)
        self._modeling_props = json['modelingProps']
        self._semantic_type_map = json['semanticTypeMap']
        self._state = ModelState(json['state'])

        # bring in the implied model types...
        self._feature_config = self._matcher.features
        self._model_type = self._matcher.model_type
        self._resampling_strategy = self._matcher.resampling_strategy
        self._num_bags = self._matcher.num_bags
        self._bag_size = self._matcher.bag_size

        return self

    def add(self, value):
        """
        Use this to add a labelled SSD or additional ontology to the local object.
        """
        if issubclass(type(value), SSD):
            self._stored = False
            self._ssds.append(value)
            return self
        elif issubclass(type(value), Ontology):
            self._stored = False
            self._ontologies.append(value)
            return self
        else:
            msg = "Only SSD or Ontologies can be added to the Octopus"
            raise ValueError(msg)

    def remove(self, value):
        """
        Use this to remove an SSD or ontology from the local object.
        """
        if issubclass(type(value), SSD):
            self._stored = False
            self._ssds.remove(value)
            return self
        elif issubclass(type(value), Ontology):
            self._stored = False
            self._ontologies.remove(value)
            return self
        else:
            msg = "Only SSD or Ontologies can be removed from the Octopus"
            raise ValueError(msg)

    def train(self):
        """
        Send the training request to the API.

        Args:
            wait : boolean indicator whether to wait for the training to finish.

        Returns: boolean -- True if model is trained, False otherwise

        """
        if not self._stored or self._session is None:
            msg = "{} is not stored on the server. Upload using <Serene>.octopii.upload()"
            raise Exception(msg)

        self._session.octopus_api.train(self.id)  # launch training

        def state():
            """Query the server for the model state"""
            json = self._session.octopus_api.item(self.id)
            octo = self.update(json,
                               self._session,
                               self._dataset_endpoint,
                               self._model_endpoint,
                               self._ontology_endpoint,
                               self._ssd_endpoint)
            return octo.state

        def is_finished():
            """Check if training is finished"""
            return state().status in {Status.COMPLETE, Status.ERROR}

        print("Training model {}...".format(self.id))
        iter = 0
        while not is_finished():
            print("\rWaiting for the training to complete " + '.'*iter, end='')
            time.sleep(0.5)  # wait in polling loop
            iter += 1

        print("Training complete for {}".format(self.id))
        logging.info("Training complete for {}.".format(self.id))
        return state().status == Status.COMPLETE

    def predict(self, dataset):
        """
        :param dataset: The dataset to perform a prediction
        :return: List of (SSD, OctopusScore), ordered by best Karma rank
        """
        if not dataset.stored:
            msg = "{} is not stored on the server.".format(dataset)
            raise ValueError(msg)

        if issubclass(type(dataset), DataSet):
            key = dataset.id
        else:
            key = int(dataset)

        blob = self._session.octopus_api.predict(self.id, key)

        prediction_list = blob['predictions']

        output = []
        for pred in prediction_list:
            ssd = SSD().update(pred['ssd'],
                               self._dataset_endpoint,
                               self._ontology_endpoint)
            score = OctopusScore(pred['score'])
            output.append(SSDResult(ssd, score))

        return output

    def matcher_predict(self, dataset, scores=True, features=False):
        """
        Returns the schema matcher results for a prediction on `dataset`

        :param dataset: The dataset to use for prediction
        :param scores: If true return the scores
        :param features: If true return all the feature values
        :return:
        """
        if not dataset.stored:
            msg = "{} is not stored on the server.".format(dataset)
            raise ValueError(msg)

        if issubclass(type(dataset), DataSet):
            key = dataset.id
        else:
            key = int(dataset)

        model = self._model_endpoint.get(self._model_id) #self._session.model_api.predict(self._model_id, key)
        return model.predict(key, scores, features)

    def __repr__(self):
        """Output string"""
        if self.stored:
            return "Octopus({}, {})".format(self.id, self.name)
        else:
            return "Octopus(local, {})".format(self.name)

    def mappings(self, data_node):
        """Returns all the Column objects that map to `datanode`..."""
        z = [(ssd.semantic_model.find(data_node), ssd) for ssd in self._ssds]

        # filter out the None values for find
        zz = [(node, ssd) for node, ssd in z if node is not None]

        return [ssd.mappings[node] for node, ssd in zz]

    @property
    def stored(self):
        return self._stored

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def date_created(self):
        return self._date_created

    @property
    def date_modified(self):
        return self._date_modified

    @property
    def id(self):
        return self._id

    @property
    def ssds(self):
        return self._ssds

    @property
    def ontologies(self):
        return self._ontologies

    @property
    def matcher_id(self):
        return self._model_id

    @property
    def matcher(self):
        return self._matcher

    @property
    def modeling_props(self):
        return self._modeling_props

    @property
    def semantic_type_map(self):
        return self._semantic_type_map

    @property
    def state(self):
        return self._state

    @property
    def feature_config(self):
        return self._feature_config

    @property
    def model_type(self):
        return self._model_type

    @property
    def resampling_strategy(self):
        return self._resampling_strategy

    @property
    def num_bags(self):
        return self._num_bags

    @property
    def bag_size(self):
        return self._bag_size


