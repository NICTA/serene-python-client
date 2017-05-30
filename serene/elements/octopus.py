"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the Octopus object
"""
import logging
import time

from ..matcher import ModelState, Status, ModelType, SamplingStrategy
from ..utils import convert_datetime, get_label, get_prefix
from .dataset import DataSet
from .semantics.ontology import Ontology
from .semantics.ssd import SSD
import networkx as nx
try:
    from math import inf
except ImportError:
    inf = float('inf')

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
        if model_type not in ModelType.values():
            msg = "Model type {} is invalid, use one of {}".format(
                model_type, ModelType.values()
            )
            raise ValueError(msg)

        if resampling_strategy not in SamplingStrategy.values():
            msg = "Resampling strategy type {} is invalid, use one of {}".format(
                resampling_strategy, SamplingStrategy.values()
            )
            raise ValueError(msg)

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
            time.sleep(3)  # wait in polling loop
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

    @staticmethod
    def convert_karma_graph(data):
        """
        convert graph.json from Karma to nx object
        :param data: json read dictionary
        :return:
        """
        logging.info("Converting karma alignment graph...")

        nodes = data["nodes"]
        links = data["links"]

        g = nx.MultiDiGraph()

        cur_id = 0
        node_map = {}
        for node in nodes:
            # filling in the nodes
            node_map[node["id"]] = cur_id
            node_data = {
                "type": "ClassNode" if node["type"] == "InternalNode" else "DataNode",
                "label": get_label(node["label"]["uri"]) if node["type"] == "InternalNode" else "",
                "lab": get_label(node["id"]) if node["type"] == "InternalNode" else "",
                "prefix": get_prefix(node["label"]["uri"]) if node["type"] == "InternalNode" else ""
            }
            g.add_node(cur_id, attr_dict=node_data)
            cur_id += 1

        cur_id = 0
        for link in links:
            # filling in the links
            source, uri, target = link["id"].split("---")
            link_data = {
                "type": link["type"],
                "weight": link["weight"],
                "label": get_label(uri),
                "prefix": get_prefix(uri)
            }
            g.add_edge(node_map[source], node_map[target], attr_dict=link_data)
            cur_id += 1
            if link["type"] in ["DataPropertyLink", "ClassInstanceLink"]:
                # change data nodes
                g.node[node_map[target]]["label"] = g.node[node_map[source]]["label"] + "---" + link_data["label"]
                g.node[node_map[target]]["lab"] = g.node[node_map[source]]["lab"] + "---" + link_data["label"]
                g.node[node_map[target]]["prefix"] = g.node[node_map[source]]["prefix"]

        logging.info("Karma alignment graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
        print("Karma alignment graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))

        return g

    def get_alignment(self):
        """Get alignment graph for octopus at position key and convert it to networkx MultiDiGraph!"""
        blob = self._session.octopus_api.alignment(self.id)
        return self.convert_karma_graph(blob)

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

    def check_modeling_props(self):
        def in_range(x, a, b, inclusive=True):
            return a <= x <= b if inclusive else a < x < b

        def prop_in_range(name, a, b, inclusive=True):
            if name not in self._modeling_props:
                return

            prop = self._modeling_props[name]

            if not in_range(prop, a, b, inclusive):
                message = "Modeling property {name} should be in range {a} {sign} {name} {sign} {b}"
                raise ValueError(message.format(
                    name=name,
                    a=a,
                    b=b,
                    sign="<=" if inclusive else "<"
                ))

        if not isinstance(self._modeling_props, dict):
            return

        prop_in_range("mappingBranchFactor", 0, inf, False)
        prop_in_range("numCandidateMappings", 0, inf, False)
        prop_in_range("topkSteinerTrees", 0, inf, False)
        prop_in_range("numSemanticTypes", 0, inf, False)
        prop_in_range("confidenceWeight", 0, 1)
        prop_in_range("coherenceWeight", 0, 1)
        prop_in_range("sizeWeight", 0, 1)
