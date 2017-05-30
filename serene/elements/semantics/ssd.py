"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the SSD dataset description object
"""
import json
import logging
import networkx as nx
import random

from collections import OrderedDict
from collections import defaultdict

from .base import BaseSemantic, DEFAULT_NS, UNKNOWN_CN, UNKNOWN_DN, ALL_CN, OBJ_PROP
from ..elements import DataProperty, ObjectProperty, SSDSearchable, SSDLink, ClassNode, DataNode
from ..elements import DataLink, ObjectLink, ClassInstanceLink, ColumnLink, SubClassLink
from ..elements import Mapping, Column, Class
from ..dataset import DataSet
from ..semantics.ontology import Ontology
from serene.utils import gen_id, convert_datetime, flatten, Searchable
from serene.visualizers import SSDVisualizer

_logger = logging.getLogger()
_logger.setLevel(logging.WARN)

class SSD(object):
    """
        Semantic source description is the translator between a DataSet
        and a set of Ontologies

        The SemanticSourceDesc contains the data columns,
        a mapping between each column to a DataNode, and a
        semantic model built from the available ontologies.
    """
    def __init__(self,
                 dataset=None,
                 ontology=None,
                 name=None):
        """
        Builds up a SemanticSourceDesc object given a dataset and
        and a parent ontology (or set of ontologies).
        """
        if dataset is not None:
            if not issubclass(type(dataset), DataSet):
                msg = "Required DataSet, not {}".format(type(dataset))
                raise Exception(msg)

            # dataset and ontology must be stored on the server!!!!
            if not dataset.stored:
                msg = "{} must be stored on the server, use <Serene>.datasets.upload"
                raise Exception(msg)

        if ontology is not None:
            if issubclass(type(ontology), Ontology):
                # we need a list of ontologies
                ontology = [ontology]

            if issubclass(type(ontology), list):
                for onto in ontology:
                    if not issubclass(type(onto), Ontology):
                        msg = "Required Ontology, not {}".format(type(onto))
                        raise Exception(msg)
                    if not onto.stored:
                        msg = "{} must be stored on the server, use <Serene>.ontologies.upload".format(onto._uri)
                        raise Exception(msg)
            else:
                msg = "Required list of ontologies, not {}".format(type(ontology))
                raise Exception(msg)

        self._name = name if name is not None else gen_id()
        self._dataset = dataset
        self._ontology = ontology if ontology is not None else []
        self._VERSION = "0.1"
        self._id = None
        self._stored = False  # is stored on the server...
        self._date_created = None
        self._date_modified = None

        # semantic model
        self._semantic_model = SSDGraph()

        # the mapping object from Column -> Mapping
        # this should always contain the complete Column set,
        # so that the user can easily see what is and isn't
        # yet mapped.
        # initialize the columns...
        if self._dataset is not None:
            for col in self._dataset.columns:
                self._semantic_model.add_node(col, index=col.id)

    def update(self, blob, dataset_endpoint, ontology_endpoint):
        """
        Create the object from json directly

        :param blob:
        :param dataset_endpoint:
        :param ontology_endpoint:
        :return:
        """
        _logger.debug("Updating ssd")

        if 'id' in blob:
            self._stored = True
            self._date_created = convert_datetime(blob['dateCreated'])
            self._date_modified = convert_datetime(blob['dateModified'])
            self._id = int(blob['id'])

        if 'name' in blob:
            self._name = blob['name']

        reader = SSDReader(blob,
                           dataset_endpoint,
                           ontology_endpoint)

        self._ontology = reader.ontology
        self._dataset = reader.dataset
        self._semantic_model = reader.semantic_model

        _logger.debug("Ssd update success!")

        return self

    def map(self, column, data_node):
        """
        Map maps a Column object to a DataNode

        :param column: The Column object in the DataSet
        :param data_node: The DataNode object to map
        :param add_class_node: Should the ClassNode in the DataNode be added
        :return:
        """
        column, data_node = self._clean_map_args(column, data_node)

        # this will raise an error if the args are not correct
        self._assert_map_args(column, data_node)

        # this now breaks from the server...
        self._stored = False

        # attempt to add the class node if not already in the model...
        if not self._semantic_model.exists(data_node.class_node, exact=True):
            self._semantic_model.add_node(data_node.class_node)

        # add the data_node
        self._semantic_model.add_node(data_node)

        # add a link between the two,
        # with namespace from data node if it's available otherwise default prefix namespace...
        if data_node.prefix:
            ns = data_node.prefix
        else:
            ns = self.default_namespace
        self._semantic_model.add_edge(
            data_node.class_node,
            data_node,
            DataLink(data_node.label,
                     prefix=ns))

        # add a link between the column and data node...
        self._semantic_model.add_edge(
            data_node,
            column,
            ColumnLink(column.name,
                       prefix=self.default_namespace))
        return self

    def link(self, src, label, dst):
        """
        Adds a link between class nodes
        :param src: The source ClassNode object
        :param label: The label for the class node
        :param dst: The destination ClassNode object
        :return:
        """
        src, dst = self._clean_link_args(src, dst)

        # this will raise an error if the args are not correct
        self._assert_link_args(src, label, dst)

        # this now breaks from the server...
        self._stored = False

        # attempt to add the class nodes if not already in the model...
        if not self._semantic_model.exists(src, exact=True):
            self._semantic_model.add_node(src)

        if not self._semantic_model.exists(dst, exact=True):
            self._semantic_model.add_node(dst)

        # add the link into
        self._semantic_model.add_edge(
            src, dst, ObjectLink(label, prefix=self.default_namespace)
        )
        return self

    def remove_link(self, item, src=None, dst=None):
        """
        Removes a link
        :param item: a link type e.g. ClassNode("Person") or ObjectLink("worksFor")
        :param src: Optional source node of link
        :param dst: Optional destination node of link
        :return:
        """
        if issubclass(type(item), str):
            item = ObjectLink(item, self.default_namespace)

        if issubclass(type(item), SSDLink):
            if item.prefix is None:
                item.prefix = self.default_namespace

            # this now breaks from the server...
            self._stored = False

            self._semantic_model.remove_edge(item, src, dst)
        else:
            msg = "Remove requires a link type"
            raise ValueError(msg)
        return self

    def remove(self, item, src=None, dst=None):
        """
        Removes nodes and links
        :param item: a node or link type e.g. ClassNode("Person") or ObjectLink("worksFor")
        :param src: Optional source node of link
        :param dst: Optional destination node of link
        :return:
        """
        if issubclass(type(item), Searchable):
            self._semantic_model.remove_node(item)
        elif issubclass(type(item), SSDLink):
            if item.prefix is None:
                item.prefix = self.default_namespace

            # this now breaks from the server...
            self._stored = False

            self._semantic_model.remove_edge(item, src, dst)
        else:
            msg = "Remove requires a node or link type"
            raise ValueError(msg)
        return self

    def _clean_link_args(self, src, dst):
        """
        Cleans the args for the link function, here a string will be automatically
        wrapped int the relevant type

        :param src: ClassNode() or str
        :param dst: ClassNode() or str
        :return: ClassNode, ClassNode
        """
        if issubclass(type(src), str):
            src = ClassNode(src, prefix=self.default_namespace)

        if issubclass(type(dst), str):
            dst = ClassNode(dst, prefix=self.default_namespace)

        if issubclass(type(src), SSDSearchable):
            if src.prefix is None:
                src = ClassNode(src.label, src.index, prefix=self.default_namespace)

        if issubclass(type(dst), SSDSearchable):
            if dst.prefix is None:
                dst = ClassNode(dst.label, dst.index, prefix=self.default_namespace)

        return src, dst

    def _clean_map_args(self, column, node):
        """
        Cleans the args for the map function, here a string will be automatically
        wrapped int the relevant type

        :param column: Column() or str
        :param node: DataNode(ClassNode(str), str), DataNode(str, str), ClassNode or str
        :return: column, node as Column(), DataNode/ClassNode
        """
        if issubclass(type(column), str):
            column = Column(column)

        if issubclass(type(node), str):
            if '.' in node:
                # this is a mapping to a data node
                items = node.split('.')
                class_node = ClassNode(items[0], prefix=self.default_namespace)
                label = items[-1]
                node = DataNode(class_node, label, prefix=self.default_namespace)
            else:
                # this is a mapping to a class node
                node = ClassNode(node, prefix=self.default_namespace)

        if issubclass(type(node), SSDSearchable):
            if node.prefix is None:
                node = DataNode(node.class_node,
                                node.label,
                                node.index,
                                self.default_namespace)

            if node.class_node.prefix is None:
                class_node = ClassNode(node.class_node.label,
                                       node.class_node.index,
                                       self.default_namespace)
                node = DataNode(class_node,
                                node.label,
                                node.index,
                                node.prefix)
        return column, node

    def _assert_map_args(self, column, data_node):
        """
        Checks that the requested Column -> DataNode mapping request is valid
        :param column:
        :param data_node:
        :return:
        """
        # Check the column status...
        if not self._column_exists(column):
            msg = "Map failed. Failed to find {} in {}".format(column, self._dataset)
            raise ValueError(msg)

        # Check the data node status...
        # first check that the class is in the ontology...
        cn = data_node.class_node
        if not self._class_node_exists(cn):
            msg = "Map failed. Failed to find {} in {}".format(cn, self._ontology)
            raise ValueError(msg)

        # next check the datanode in the ontology
        if not self._data_node_exists(data_node):
            msg = "Map failed. Failed to find {} in {}".format(data_node, self._ontology)
            raise ValueError(msg)

        # for columns, only one map can exist...
        if self._semantic_model.degree(column) > 0:
            msg = "Map failed. {} already has an existing mapping".format(column)
            raise ValueError(msg)

        # next check the datanode is unique
        if self._semantic_model.exists(data_node, exact=True):
            msg = "Map failed. {} already exists in the SSD".format(data_node)
            raise ValueError(msg)

    def _assert_link_args(self, src, label, dst):
        """
        Checks that the requested Class - Class link request is valid
        :param src: The source class node
        :param label: The label name for the link
        :param dst: The destination class node
        :return: None
        """
        # first check the src class in the ontology
        if not self._class_node_exists(src):
            msg = "Link failed. Failed to find {} in {}".format(src, self._ontology)
            raise ValueError(msg)

        # first check the dst class in the ontology
        if not self._class_node_exists(dst):
            msg = "Link failed. Failed to find {} in {}".format(dst, self._ontology)
            raise ValueError(msg)

        # next check the link exists in the ontology
        if not self._link_exists(src, label, dst):
            msg = "Link failed. Failed to find {}-{}-{} in {}".format(src, label, dst, self._ontology)
            raise ValueError(msg)

    def _link_exists(self, src, label, dst):
        """
        Checks to see that the link exists in the ontology
        :param src:
        :param label:
        :param dst:
        :return:
        """
        _logger.debug("Searching for {}-{}-{}".format(src, label, dst))

        link_target = ObjectProperty(label, Class(src.label), Class(dst.label))

        _logger.debug("using target: {}".format(link_target))

        link = None
        for onto in self._ontology:
            # find any match in all ontologies
            try:
                link = ObjectProperty.search(onto.ilinks, link_target)
            except:
                continue
            break

        return link is not None

    def _column_exists(self, column):
        """
        Check that this column exists in the dataset.
        :param column:
        :return:
        """
        # Check the column status...
        col = Column.search(self._dataset.columns, column)
        return col is not None

    def _class_node_exists(self, cn: ClassNode):
        """
        Check that the class node exists in the defined ontology
        :param cn: The ClassNode parameter
        :return:
        """
        # Check the dataset status...
        # first check the class
        cls_target = Class(cn.label, prefix=cn.prefix)
        cls = None
        for onto in self._ontology:
            # find any match in all ontologies
            try:
                cls = Class.search(onto.class_nodes, cls_target)
                if cls is not None:
                    break
            except:
                continue

        return cls is not None

    def _data_node_exists(self, data_node: DataNode):
        """
        Check that the data node exists in the defined ontology
        :param data_node: A DataNode object to check
        :return:
        """
        # next check the datanode
        cn = data_node.class_node

        # construct a search param for the ontology Class and DataNode...
        cls_target = Class(cn.label, prefix=cn.prefix)
        target = DataProperty(cls_target, data_node.label)

        dn = None
        for onto in self._ontology:
            # find any match in all ontologies
            try:
                dn = DataProperty.search(onto.idata_nodes, target)
                if dn is not None:
                    break
            except:
                continue

        return dn is not None

    @property
    def default_namespace(self):
        """
        The default namespace to add to ClassNode, DataNode and Links if missing.
        If there are ontologies, it will return the first not empty namespace of the ontology;
        if all ontology namespaces are empty, it will return the default namespace.
        An error will be raised if there are no ontologies.
        :return: str
        """
        # TODO: check if it's ok to take just the first ontology from the list...
        if self._ontology and len(self._ontology):
            for onto in self._ontology:
                if onto.namespace:
                    return onto.namespace
            return DEFAULT_NS
        else:
            msg = "No ontology available in SSD."
            raise Exception(msg)

    def fill_unknown(self):
        """
        Make all unmapped columns from the dataset as instances of unknown.
        Then add Thing/All node and make all class nodes as subclass of Thing/ connect to All.
        This way we ensure that ssd is connected.
        Use with caution!
        """
        unmapped_cols = [col for col in self.dataset.columns if col not in self.columns]
        # TODO: check that Unknown is among ontologies
        for i, col in enumerate(unmapped_cols):
            self._semantic_model.add_node(col, index=col.id)
            data_node = DataNode(ClassNode(UNKNOWN_CN, prefix=DEFAULT_NS), label=UNKNOWN_DN,
                                 index=i,
                                 prefix=DEFAULT_NS)
            self.map(col, data_node)
        if len(unmapped_cols):
            self._add_all_node()
        return self

    def _add_all_node(self):
        """
        Add All node to ssd and make all class nodes connect to All.
        Helper method for fill_unknown
        """
        class_nodes = self.class_nodes
        # TODO: check that All is among ontologies
        thing_node = ClassNode(ALL_CN, prefix=DEFAULT_NS)
        self._semantic_model.add_node(thing_node)
        for cn in class_nodes:
            self._semantic_model.add_edge(
                thing_node, cn, ObjectLink(OBJ_PROP, prefix=DEFAULT_NS))
        return self

    def _add_thing_node(self):
        """
        Add Thing node to ssd and make all class nodes subclass of Thing.
        Helper method for fill_unknown
        """
        class_nodes = self.class_nodes
        prefix = "http://www.w3.org/2000/01/rdf-schema#"
        thing_node = ClassNode("Thing", prefix="http://www.w3.org/2002/07/owl#")
        self._semantic_model.add_node(thing_node)
        for cn in class_nodes:
            self._semantic_model.add_edge(
                cn, thing_node, SubClassLink("subClassOf", prefix=prefix))
        return self

    def __repr__(self):
        """Output string"""
        if self.stored:
            return "SSD({}, {})".format(self.id, self.name)
        else:
            return "SSD(local, {})".format(self.name)

    @property
    def class_nodes(self):
        return self._semantic_model.class_nodes

    @property
    def data_nodes(self):
        return self._semantic_model.data_nodes

    @property
    def columns(self):
        return self._semantic_model.columns  # self._dataset.columns

    @property
    def data_links(self):
        return self._semantic_model.data_links

    @property
    def object_links(self):
        return self._semantic_model.object_links

    @property
    def links(self):
        return self.data_links + self.object_links

    @property
    def dataset(self):
        return self._dataset

    @property
    def ontology(self):
        return self._ontology

    @property
    def mappings(self):
        return self._semantic_model.mappings

    @property
    def unmapped_columns(self):
        mapped_cols = set(self.mappings.values())
        all_cols = set(self._dataset.columns)
        return all_cols - mapped_cols

    @property
    def semantic_model(self):
        return self._semantic_model

    @property
    def json(self):
        return SSDJsonWriter(self).to_json()

    @property
    def json_dict(self):
        return SSDJsonWriter(self).to_dict()

    @property
    def stored(self):
        return self._stored

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    def show(self, title='', outfile=None):
        """Displays the SSD"""
        self._semantic_model.show(name=self._dataset.filename, title=title, outfile=outfile)

    def __repr__(self):
        props = "name={}, dataset={}, ontology={}".format(
            self._name, self._dataset, self._ontology)
        if self._stored:
            return "SSD({}, {})".format(self._id, props)
        else:
            return "SSD(local, {})".format(props)

    def evaluate(self, ground_truth, include_all=False, include_cols=True):
        """
        Evaluate this ssd against ground truth
        :param ground_truth: another ssd
        :param include_all: boolean whether to include the connector node All
        :param include_cols: boolean wheter to include column links
        :return:
        """
        logging.info("Calculating comparison metrics for ssds")
        ssd_triples = self.get_triples(include_all=include_all, include_cols=include_cols)
        ground_triples = ground_truth.get_triples(include_all=include_all, include_cols=include_cols)
        comparison = {"precision": self.get_precision(ground_triples, ssd_triples),
                      "recall": self.get_recall(ground_triples, ssd_triples),
                      "jaccard": self.get_jaccard(ground_triples, ssd_triples)}
        return comparison

    def _node_uri(self, node_id):
        """Get uri of the node in ssd"""
        node = self._semantic_model._graph.node[node_id]["data"]
        if type(node) == Column:
            # for columns we return its name
            return node.name
        return node.prefix + node.label

    @staticmethod
    def _link_uri(ld):
        """Get uri of link"""
        if type(ld) == ColumnLink:
            return "ColumnLink"
        else:
            return ld.prefix + ld.label

    def get_triples(self, include_all=False, include_cols=True):
        """
        Get extended RDF triples from the ssd including columns
        - include mappings to columns
        - include All unknown
        :param include_all: boolean whether to include the connector node All
        :param include_cols: boolean wheter to include column links
        """
        # logging.info("Extracting rdf triples from ssd...")
        logging.info("Extracting rdf triples from ssd...")
        triples = []
        # a triple = (uri of subject, uri of predicate, uri of object, num)
        # num just enumerates how many times the same triple appears in the semantic model
        triple_count = defaultdict(int)
        links = self._semantic_model._graph.edges(data=True)
        all_uri = DEFAULT_NS + ALL_CN
        for l_start, l_end, ld in links:
            pred = self._link_uri(ld["data"])
            if not (include_cols) and pred == "ColumnLink":
                continue

            subj, obj = self._node_uri(l_start), self._node_uri(l_end)
            if not (include_all) and (subj == all_uri or obj == all_uri):
                continue
            triple = (subj, pred, obj)
            triples.append(triple + (triple_count[triple],))
            # increase the counter for this triple
            triple_count[triple] += 1

        # logging.info("Extracted {} rdf triples from ssd.".format(len(triples)))
        logging.info("Extracted {} extended rdf triples from ssd.".format(len(triples)))
        triples = set(triples)
        logging.info("Extracted {} unique rdf triples from ssd.".format(len(triples)))

        return triples

    @staticmethod
    def get_precision(correct_triples, result_triples):
        """
        Calculate precision by comparing triples of the ground truth with the predicted result.
        :param correct_triples: set
        :param result_triples: set
        :return:
        """
        intersection = float(len(correct_triples & result_triples))
        result_size = len(result_triples)
        if result_size:
            return intersection / result_size
        return 0.0

    @staticmethod
    def get_recall(correct_triples, result_triples):
        """
        Calculate recall by comparing triples of the ground truth with the predicted result.
        :param correct_triples: set
        :param result_triples: set
        :return:
        """
        intersection = float(len(correct_triples & result_triples))
        correct_size = len(correct_triples)
        if correct_size:
            return intersection / correct_size
        return 0.0

    @staticmethod
    def get_jaccard(correct_triples, result_triples):
        """
        Calculate jaccard coefficient by comparing triples of the ground truth with the predicted result.
        :param correct_triples: set
        :param result_triples: set
        :return:
        """
        intersection = float(len(correct_triples & result_triples))
        union_size = len(correct_triples | result_triples)
        if union_size:
            return intersection / union_size
        return 0.0


class SSDGraph(object):
    """
    The Semantic Model object for the SSD
    """
    def __init__(self):
        """
        Simple initialization to start the id counters and
        hold the graph object
        """
        self._node_id = 0
        self._edge_id = 0
        self._graph = nx.MultiDiGraph() # holds the ClassNodes, DataNodes and Column objects
        self._lookup = {}  # object -> int key
        self.DATA_KEY = 'data'

    def _type_list(self, value_type):
        """
        Helper function to filter out the lookup keys by type.

        :param value_type: The type of the node to search.
        :return:
        """
        return [x for x in self._lookup.keys() if type(x) == value_type]

    def find(self, node: Searchable, exact=False) -> Searchable:
        """
        Helper function to find `node` in the semantic model. The search
        is loose on additional node parameters other than label.

        :param node: A ClassNode or DataNode
        :param exact: Should find return an exact .eq. match or just approximate through the type 'getters'
        :return:
        """
        # first collect the candidates from the lookup keys...
        candidates = self._type_list(type(node))
        return type(node).search(candidates, node, exact=exact)

    def exists(self, node: Searchable, exact=False) -> bool:
        """
        Helper function to check if a node exists...
        :param node: A DataNode or ClassNode
        :param exact: Should the match be exact or approximate
        :return:
        """
        return self.find(node, exact) is not None

    def degree(self, node: Searchable):
        """Returns the degree of a node in the graph"""
        key = self.find(node)
        if key is None:
            return 0
        return self._graph.degree(self._lookup[key])

    def add_node(self, node: Searchable, index=None):
        """
        Adds a node into the semantic model

        :param node: A DataNode or ClassNode to add into the graph
        :param index: The override index parameter. If None the auto-increment will be used.
        :return:
        """
        true_node = self.find(node, exact=True)

        if true_node is not None:
            msg = "{} already exists in the SSD: {}".format(node, true_node)
            _logger.debug(msg)
            # keep the same index in this case...
            index = self._lookup[true_node]

        # set the index
        if index is None:
            # we need to be careful that this node is not in the graph...
            nodes = self._graph.nodes()
            index = max(nodes) + 1 if len(nodes) else 0

        # add the node into the semantic model
        self._graph.add_node(index, data=node, node_id=index)

        # add this node into the lookup...
        self._lookup[node] = index

        return index

    def add_edge(self,
                 src: SSDSearchable,
                 dst: SSDSearchable,
                 link: SSDLink,
                 index=None):
        """
        Adds an edge from `src` -> `dest` with `link`
        :param src: A SSDSearchable e.g. a DataNode or ClassNode
        :param dst: A DataNode or ClassNode
        :param link: An SSDLink value to store on the edge.
        :param index: The override index parameter. If None the auto-increment will be used.
        :return:
        """
        if type(src) == int and type(dst) == int:
            i_s = src
            i_d = dst
        else:
            true_src, true_dst = self._check_edge_args(src, dst)

            # make sure we aren't rewriting...
            i_s = self._lookup[true_src]
            i_d = self._lookup[true_dst]

        if link in self.all_edge_data(i_s, i_d):
            _logger.debug("{} is already in the SSD")
            return

        # set the index...
        if index is None:
            # we need to be careful that the auto index is not in the graph...
            edges = [v['edge_id'] for _, _, v in self._graph.edges(data=True)]
            index = max(edges) + 1 if len(edges) else 0

        if i_s not in self._graph.node:
            msg = "Link failed. Could not find source node {} in semantic model".format(i_s)
            raise Exception(msg)

        if i_d not in self._graph.node:
            msg = "Link failed. Could not find destination node {} in semantic model".format(i_d)
            raise Exception(msg)

        self._graph.add_edge(i_s,
                             i_d,
                             data=link,
                             edge_id=index)
        return i_s, i_d

    def all_neighbors(self, n):
        """returns all neighbours of node `n`"""
        return self._graph.successors(n) + self._graph.predecessors(n)

    def remove_node(self, node: Searchable):
        """
        Removes a node from the graph. Note that a Column node
        cannot be removed. If node is a Column, the mapping will
        be removed instead

        :param node: A DataNode, Column or ClassNode
        :return: None
        """
        true_node = self.find(node)
        if true_node is None:
            msg = "Remove node failed. {} does not exist.".format(true_node)
            raise ValueError(msg)

        key = self._lookup[true_node]

        if issubclass(type(true_node), Column):
            # if it's a column, we just remove the data node...
            for dn in self.all_neighbors(key):
                for v in self._ilookup[dn]:
                    self.remove_node(v)
        else:
            # otherwise, we can remove the node and it's
            # adjacent links. We also make sure that any
            # hanging nodes are removed too...
            neighbors = self.all_neighbors(key)
            self._graph.remove_node(key)
            del self._lookup[true_node]

            # remove any hanging neighbors...
            m = self._ilookup
            for n in neighbors:
                # if there are any nodes with no outgoing links
                # then we should remove them.
                if self._graph.out_degree(n) == 0:
                    # we remove the node by the item to chain
                    # the removal...
                    for item in m[n]:
                        if not issubclass(type(item), Column):
                            self.remove_node(item)
        return self

    @property
    def _ilookup(self):
        """inverse lookup..."""
        m = defaultdict(list)
        for k, v in self._lookup.items():
            m[v].append(k)
        return m

    def remove_edge(self,
                    item: SSDLink,
                    src: Searchable=None,
                    dst: Searchable=None):
        """
        Removes an edge from the graph. Note that a Column node
        cannot be removed. If node is a Column, the mapping will
        be removed instead

        :param item: The link object
        :param src: A DataNode, Column or ClassNode
        :param dst: A DataNode, Column or ClassNode
        :return: None
        """
        edges = [(x, y, k) for x, y, k, z in self._edge_list if z == item]

        if len(edges) > 1:
            # ambiguous, may need the src/dst to find it
            if src is not None and dst is not None:
                true_src, true_dst = self._check_edge_args(src, dst)

                i = self._lookup[true_src]
                j = self._lookup[true_dst]

                if len(self._graph.edge[i][j]) == 1:
                    # remove the edge and its adjacent links if only one
                    self._graph.remove_edge(true_src, true_dst, 0)

            msg = "Failed to remove edge. {} is ambiguous".format(item)
            raise ValueError(msg)
        elif len(edges) == 0:
            msg = "Failed to remove edge. {} could not be found.".format(item)
            raise ValueError(msg)
        else:
            x, y, k = edges[0]
            self._graph.remove_edge(x, y, k)

            if not len(self.all_neighbors(x)):
                for v in self._ilookup[x]:
                    self.remove_node(v)

            if not len(self.all_neighbors(y)):
                for v in self._ilookup[y]:
                    self.remove_node(v)

        return self

    def show(self, name='source', title='',  outfile=None):
        """
        Prints to the screen and also shows a graphviz visualization.

        :param title: Optional title for the graph
        :param outfile: file where graph will be stored; if None - a temporary file will be created
        :return:
        """
        SSDVisualizer(self, name=name, outfile=outfile).show(title=title)

    @property
    def graph(self):
        """Reference to the graph object"""
        return self._graph

    def node_data(self, node):
        """Helper function to return the data stored at the node"""
        return self._graph.node[node][self.DATA_KEY]

    def edge_data(self, src, dst, index=0):
        """Helper function to return the data stored on the edge"""
        return self._graph.edge[src][dst][index][self.DATA_KEY]

    def all_edge_data(self, src, dst):
        """Helper function to return all the data stored on the edge"""
        if src in self._graph.edge and dst in self._graph.edge[src]:

            z = self._graph.edge[src][dst]
            return [z[i][self.DATA_KEY] for i in range(len(z))
                    if z[i] is not None and
                    z[i][self.DATA_KEY] is not None]
        else:
            return []

    @property
    def class_nodes(self):
        """Returns the ClassNode objects in the semantic model"""
        return [self.node_data(n) for n in self._graph.nodes()
                if type(self.node_data(n)) == ClassNode]

    @property
    def data_nodes(self):
        """Returns the DataNode objects in the graph"""
        return [self.node_data(n) for n in self._graph.nodes()
                if type(self.node_data(n)) == DataNode]

    @property
    def columns(self):
        """Returns the Column objects in the graph"""
        return [self.node_data(n) for n in self._graph.nodes()
                if type(self.node_data(n)) == Column]

    @property
    def _edge_list(self):
        """helper function to return 4-tuple of (x, y, key, item)"""
        edge_dicts = [(e1, e2, self._graph.edge[e1][e2]) for e1, e2 in self._graph.edges()]

        # flatten out to (e1, e2, key, value)
        arr = []
        for x, y, z in edge_dicts:
            for k, v in z.items():
                arr.append((x, y, k, v[self.DATA_KEY]))

        return arr

    def _check_edge_args(self, src, dst):
        """
        Checks that the edges exist...

        :param src: A DataNode, Column or ClassNode
        :param dst: A DataNode, Column or ClassNode
        :return: src, dst
        """
        true_src = self.find(src)
        true_dst = self.find(dst)

        if true_src is None:
            msg = "Link source {} does not exist in the SSD".format(src)
            raise Exception(msg)

        if true_dst is None:
            msg = "Link dest {} does not exist in the SSD".format(dst)
            raise Exception(msg)

        return true_src, true_dst

    def _edges_data(self):
        """
        Helper function to extract all the attribute maps. Note that a multigraph has
        a dictionary at each edge[x][y] point, so we need to pull out each edge.
        """
        return flatten(self._graph.edge[e1][e2].values() for e1, e2 in self._graph.edges())

    def _edge_data_filter(self, value_type: SSDLink):
        """
        Helper function to extract a single type of edge...
        :param value_type: The edge type to filter on
        :return: List of SSDLink objects
        """
        return [e[self.DATA_KEY] for e in self._edges_data()
                if type(e[self.DATA_KEY]) == value_type]

    @property
    def data_links(self):
        return self._edge_data_filter(DataLink)

    @property
    def object_links(self):
        return self._edge_data_filter(ObjectLink)

    @property
    def class_links(self):
        return self._edge_data_filter(ClassInstanceLink)

    @property
    def column_links(self):
        return self._edge_data_filter(ColumnLink)

    @property
    def mappings(self):
        """Helper function for the column -> data mappings"""
        def data(edge):
            return self._graph.node[edge][self.DATA_KEY]

        return {data(e1): data(e2) for e1, e2, _, link
                in self._edge_list if type(link) == ColumnLink}


class SSDReader(object):
    """
    The SSDReader is a helper object used to parse an SSD json
    blob from the server.
    """
    def __init__(self, blob, dataset_endpoint, ontology_endpoint):
        """Builds up the relevant properties from the json blob `json`
            note that we need references to the endpoints to ensure
            that the information is up-to-date with the server.
        """
        self._on_endpoint = ontology_endpoint
        self._ds_endpoint = dataset_endpoint

        # add the ontology and dataset objects
        self._ontology = self._find_ontology(blob)
        self._dataset = self._find_dataset(blob)

        if len(self._ontology):
            self._ns = self._ontology[0].namespace
        else:
            msg = "Failed to initialize SSD reader. No ontology found."
            raise Exception(msg)

        # build up the semantic model graph...
        self._graph = SSDGraph()
        self._column_map = self._ds_endpoint.columns
        self._build_graph(blob)

    def _build_graph(self, blob):
        """Builds up the semantic model graph from the JSON blob"""
        _logger.debug("Building graph for semantic model from blob")
        nodes = blob['semanticModel']['nodes']
        links = blob['semanticModel']['links']
        mappings = blob['mappings']

        self._build_graph_nodes(nodes)
        self._build_graph_links(links)
        self._build_data_nodes(links)
        self._build_graph_mappings(mappings)

    def _build_graph_nodes(self, nodes):
        """Pulls out the node objects and adds them into the graph using the interface"""
        _logger.debug("Building nodes for semantic model from blob")
        cls_table = defaultdict(list)

        for obj in nodes:
            index = obj['id']
            prefix = obj['prefix'] if 'prefix' in obj else self._ns
            label = obj['label']

            if obj['type'] == "ClassNode":
                item = ClassNode(label, prefix=prefix)
                cls_table[item].append(index)
                # self._graph.add_node(item, index=index)

        for item, values in cls_table.items():
            if len(values) == 1:
                self._graph.add_node(item, index=values[0])
            else:
                for i, v in enumerate(values):
                    new = ClassNode(item.label, index=i, prefix=item.prefix)
                    self._graph.add_node(new, index=v)

    def _build_graph_links(self, links):
        """
        Pulls out the link objects and adds them into the graph using the interface.
        Here we consider ObjectPropertyLink and CompactSubClassLink.
        """
        _logger.debug("Building links for semantic model from blob")
        for obj in links:
            index = obj['id']
            src = obj['source']
            dst = obj['target']
            label = obj['label']
            prefix = obj['prefix'] if 'prefix' in obj else self._ns

            if obj['type'] == "ObjectPropertyLink":
                item = ObjectLink(label, prefix=prefix)
                self._graph.add_edge(src, dst, item, index=index)
            elif obj['type'] == "SubClassLink":
                item = SubClassLink(label, prefix=prefix)
                self._graph.add_edge(src, dst, item, index=index)

    def _build_data_nodes(self, links):
        """Pulls out the data nodes from the property links and adds them"""
        _logger.debug("Building data nodes for semantic model from blob")
        dn_table = defaultdict(list)

        # first collect the links and add them to the table...
        for obj in links:
            index = obj['id']
            src = obj['source']
            dst = obj['target']
            label = obj['label']
            prefix = obj['prefix'] if 'prefix' in obj else self._ns

            if obj['type'] in {"DataPropertyLink", "ClassInstanceLink"}:
                # first we create the data node...
                class_node = self._graph.graph.node[src]['data']
                data_node = DataNode(class_node, label, prefix=prefix)
                dn_table[data_node].append((src, dst, index, obj['type']))

        # now we use the data node table to construct the
        # nodes in order...
        self._build_data_nodes_ordered(dn_table)

    def _build_data_nodes_ordered(self, dn_table):
        """
        Helper function to create the data nodes in order. We need
        to do it this way to accommodate multiple data nodes mapping
        onto a single class node.

        :param dn_table: Dict of DataNode() -> [(class_id, node_id, link_index)...]
        """
        _logger.debug("Building ordered nodes for semantic model from blob")
        # We need to add the nodes from the dn_table....
        for node, values in dn_table.items():
            if len(values) == 1:
                src, dst, index, obj_type = values[0]
                # this case is easy, just add the node and link...
                self._graph.add_node(node, index=dst)
                self._add_graph_link(node, obj_type, src, dst, index)
            else:
                # in this case, we need to provide an index...
                for i, v in enumerate(values):
                    src, dst, index, obj_type = v
                    item = DataNode(node.class_node,
                                    node.label,
                                    index=i,
                                    prefix=node.prefix)

                    self._graph.add_node(item, index=dst)
                    self._add_graph_link(node, obj_type, src, dst, index)

    def _add_graph_link(self, node, obj_type, src, dst, index):
        """
        Helper function to add a link to the semantic model graph.
        :param node:
        :param obj_type:
        :param src:
        :param dst:
        :param index:
        :return:
        """
        # next we add the link between the class
        if obj_type == "DataPropertyLink":
            item = DataLink(node.label, prefix=node.prefix)
        else:
            item = ClassInstanceLink(node.label, prefix=node.prefix)

        try:
            self._graph.add_edge(src, dst, item, index=index)
        except Exception as e:
            raise Exception("Failed to add link {}: {}".format(item, str(e)))

    def _build_graph_mappings(self, mappings):
        """
        Pulls out the mapping code and adds them into the graph. Note that we need to
        add in the columns as nodes first...
        """
        for obj in mappings:
            dst = obj['attribute']
            src = obj['node']
            column = self._column_map[dst]

            # add the column to the graph...
            col_id = self._graph.add_node(column, column.id)

            # add the link to the column...
            item = ColumnLink(column.name)
            try:
                self._graph.add_edge(src, col_id, item)
            except Exception as e:
                raise Exception("Failed to add column link {}: {}".format(item, str(e)))

    @property
    def ontology(self):
        return self._ontology

    @property
    def dataset(self):
        return self._dataset

    @property
    def semantic_model(self):
        return self._graph

    def _find_ontology(self, json):
        """Pulls the ontology reference from the SSD and queries the server"""
        ontologies = json['ontologies']
        return [self._on_endpoint.get(onto) for onto in ontologies]

    def _find_dataset(self, json):
        """Attempts to grab the dataset out from the json string"""

        columns = [c['attribute'] for c in json['mappings']]

        if not len(columns):
            msg = "No columns present in ssd file mappings."
            raise Exception(msg)

        col_map = self._ds_endpoint.columns

        if columns[0] not in col_map:
            msg = "Column {} does not appear on the server".format(columns[0])
            raise Exception(msg)

        ds_key = col_map[columns[0]].datasetID

        return self._ds_endpoint.get(ds_key)


class SSDJsonWriter(object):
    """
    Helper class to build up the json output for the SSD file.
    """
    def __init__(self, ssd):
        """

        :param ssd:
        """
        self._ssd = ssd

    def to_dict(self):
        """Builds the dictionary representation of the SSD"""
        d = OrderedDict()
        d["name"] = self._ssd.name
        d["ontologies"] = [onto.id for onto in self._ssd.ontology]
        d["semanticModel"] = self.semantic_model
        d["mappings"] = self.mappings
        return d

    def to_json(self):
        """Builds the complete json object string"""
        return json.dumps(self.to_dict())

    @property
    def _json_links(self):
        """
        Builds the links section for the semantic model
        :return:
        """
        links = [(src, dst, item) for src, dst, item
                 in self._ssd.semantic_model.graph.edges(data=True)
                 if type(item['data']) != ColumnLink]

        return [{"id": item['edge_id'],
                 "source": src,
                 "target": dst,
                 "label": item['data'].label,
                 "type": item['data'].type,
                 "prefix": str(item['data'].prefix)} for src, dst, item in links]

    @property
    def _json_nodes(self):
        """
        Builds the nodes section for the semantic model
        :return:
        """
        nodes = [(index, value) for index, value
                 in self._ssd.semantic_model.graph.nodes(data=True)
                 if (type(value['data']) == DataNode or type(value['data']) == ClassNode)]

        return [{"id": index,
                 "label": value['data'].full_label,
                 "type": value['data'].type,
                 "prefix": str(value['data'].prefix)} for index, value in nodes]

    @property
    def semantic_model(self):
        """Builds out the .ssd semantic model section..."""
        return {
            "nodes": self._json_nodes,
            "links": self._json_links
        }

    @property
    def mappings(self):
        """Builds out the .ssd mapping section.."""
        maps = [(src, dst) for src, dst, data
                in self._ssd.semantic_model.graph.edges(data=True)
                if type(data['data']) == ColumnLink]

        return [{
                    "attribute": dst,
                    "node": src
                } for src, dst in maps]
