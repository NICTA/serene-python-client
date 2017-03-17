"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the SSD dataset description object
"""
import json
import logging
import networkx as nx

from collections import OrderedDict
from collections import defaultdict

from .base import BaseSemantic
from ..elements import DataProperty, ObjectProperty, SSDSearchable, SSDLink, ClassNode, DataNode
from ..elements import DataLink, ObjectLink, ClassInstanceLink
from ..elements import Mapping, Column, Class
from ..dataset import DataSet
from ..semantics.ontology import Ontology
from serene.utils import gen_id, convert_datetime, flatten
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
            if not issubclass(type(ontology), Ontology):
                msg = "Required Ontology, not {}".format(type(ontology))
                raise Exception(msg)

            if not ontology.stored:
                msg = "{} must be stored on the server, use <Serene>.ontologies.upload"
                raise Exception(msg)

        self._name = name if name is not None else gen_id()
        self._dataset = dataset
        self._ontology = ontology
        self._VERSION = "0.1"
        self._id = None
        self._stored = False  # is stored on the server...
        self._date_created = None
        self._date_modified = None

        # semantic model
        #self._semantic_model = BaseSemantic()

        self._graph = SSDGraph() #nx.MultiDiGraph()


        # the mapping object from Column -> Mapping
        # this should always contain the complete Column set,
        # so that the user can easily see what is and isn't
        # yet mapped.
        if self._dataset is not None:
            self._mapping = self._init_mapping()
        else:
            self._mapping = {}

    def _init_mapping(self):
        """Initialize the mapping to hold Column -> None """
        m = {}
        for i, column in enumerate(self._dataset.columns):
            m[column] = Mapping(column, node=None)
        return m

    def update(self, blob, dataset_endpoint, ontology_endpoint):
        """
        Create the object from json directly

        :param blob:
        :param dataset_endpoint:
        :param ontology_endpoint:
        :return:
        """
        if 'id' in blob:
            self._stored = True
            self._date_created = convert_datetime(blob['dateCreated'])
            self._date_modified = convert_datetime(blob['dateModified'])
            self._id = int(blob['id'])

        if 'name' in blob:
            self._name = blob['name']

        reader = SSDReader(blob, dataset_endpoint, ontology_endpoint)

        self._ontology = reader.ontology
        self._dataset = reader.dataset
        #self._semantic_model = reader.semantic_model

        # build up the mapping through the interface...
        #for col, ds in reader.mapping:
        #    self.map(col, ds, add_class_node=False)

        # add the class nodes

        # add the data nodes

        # add the mappings

        # add the links

        return self

    def map(self, column, data_node):
        """
        Map maps a Column object to a DataNode

        :param column: The Column object in the DataSet
        :param data_node: The DataNode object to map
        :param add_class_node: Should the ClassNode in the DataNode be added
        :return:
        """
        column, data_node = self._clean_args(column, data_node)

        # this will raise an error if the args are not correct
        self._assert_map_args(column, data_node)

        # attempt to add the class node
        self._graph.add_node(data_node.class_node)

        # add the data node...
        self._graph.add_node(data_node)

        # add a link between the two, with default prefix namespace...
        self._graph.add_edge(data_node.class_node,
                             data_node,
                             DataLink(data_node.label,
                                      prefix=self._ontology.uri))

        return self

    @staticmethod
    def _clean_args(column, node):
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
                node = DataNode(*node.split('.'))
            else:
                node = ClassNode(node)

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
            msg = "Failed to find {} in {}".format(column, self._dataset)
            raise ValueError(msg)

        # Check the data node status...
        # first check the class
        cn = data_node.class_node
        if not self._class_node_exists(cn):
            msg = "Failed to find {} in {}".format(cn, self._ontology)
            raise ValueError(msg)

        # next check the datanode
        if not self._data_node_exists(data_node):
            msg = "Failed to find {} in {}".format(data_node, self._ontology)
            raise ValueError(msg)

    def _column_exists(self, column):
        """

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
        cls = Class.search(self._ontology.class_nodes, cls_target)

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

        dn = DataProperty.search(self._ontology.idata_nodes, target)

        return dn is not None

    @property
    def class_nodes(self):
        return self._graph.class_nodes

    @property
    def data_nodes(self):
        return self._graph.data_nodes

    @property
    def columns(self):
        return self._dataset.columns

    @property
    def data_links(self):
        return self._graph.data_links

    @property
    def object_links(self):
        return self._graph.object_links

    @property
    def links(self):
        return self.data_links + self.object_links

    @property
    def dataset(self):
        return self._dataset

    @property
    def ontology(self):
        return self._ontology


class SSDGraph(object):
    """

    """
    def __init__(self):
        """

        """
        self._node_id = 0
        self._edge_id = 0
        self._graph = nx.MultiDiGraph()
        self._lookup = {}
        self.DATA_KEY = 'data'

    def _type_list(self, value_type):
        """
        Helper function to filter out the lookup keys by type.

        :param value_type: The type of the node to search.
        :return:
        """
        return [x for x in self._lookup.keys() if type(x) == value_type]

    def find(self, node: SSDSearchable):
        """
        Helper function to find `node` in the semantic model. The search
        is loose on additional node parameters other than label.

        :param node: A ClassNode or DataNode
        :return:
        """
        # first collect the candidates from the lookup keys...
        candidates = self._type_list(type(node))
        return type(node).search(candidates, node)

    def exists(self, node: SSDSearchable):
        """
        Helper function to check if a node exists...
        :param node: A DataNode or ClassNode
        :return:
        """
        return self.find(node) is not None

    def add_node(self, node: SSDSearchable, index=None):
        """
        Adds a node into the semantic model

        :param node: A DataNode or ClassNode to add into the graph
        :param index: The override index parameter. If None the auto-increment will be used.
        :return:
        """
        n = self.find(node)
        if n is not None:
            msg = "{} already exists in the SSD: {}".format(node, n)
            raise Exception(msg)

        # set the index
        if index is None:
            index = self._node_id
            self._node_id += 1

        # add the node into the semantic model
        self._graph.add_node(index, data=node, node_id=index)

        # add this node into the lookup...
        self._lookup[node] = index

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
        true_src = self.find(src)
        true_dst = self.find(dst)

        if true_src is None:
            msg = "Link source {} does not exist in the SSD".format(src)
            raise Exception(msg)

        if true_dst is None:
            msg = "Link dest {} does not exist in the SSD".format(dst)
            raise Exception(msg)

        # set the index...
        if index is None:
            index = self._edge_id
            self._edge_id += 1

        self._graph.add_edge(self._lookup[true_src],
                             self._lookup[true_dst],
                             data=link,
                             edge_id=index)

    def _node_data(self, node):
        """Helper function to return the data stored at the node"""
        return self._graph.node[node][self.DATA_KEY]

    def _edge_data(self, src, dst, index=0):
        """Helper function to return the data stored on the edge"""
        return self._graph.edge[src][dst][index][self.DATA_KEY]

    @property
    def class_nodes(self):
        """Returns the ClassNode objects in the semantic model"""
        return [self._node_data(n) for n in self._graph.nodes()
                if type(self._node_data(n)) == ClassNode]

    @property
    def data_nodes(self):
        """Returns the DataNode objects in the graph"""
        return [self._node_data(n) for n in self._graph.nodes()
                if type(self._node_data(n)) == DataNode]

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
        :return:
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
    def object_links(self):
        return self._edge_data_filter(ClassInstanceLink)

            # """
    #     Semantic source description is the translator between a DataSet
    #     and a set of Ontologies
    #
    #     The SemanticSourceDesc contains the data columns,
    #     a mapping between each column to a DataNode, and a
    #     semantic model built from the available ontologies.
    # """
    # def __init__(self,
    #              dataset=None,
    #              ontology=None,
    #              name=None):
    #     """
    #     Builds up a SemanticSourceDesc object given a dataset and
    #     and a parent ontology (or set of ontologies).
    #     """
    #     if dataset is not None:
    #         if not issubclass(type(dataset), DataSet):
    #             msg = "Required DataSet, not {}".format(type(dataset))
    #             raise Exception(msg)
    #
    #         # dataset and ontology must be stored on the server!!!!
    #         if not dataset.stored:
    #             msg = "{} must be stored on the server, use <Serene>.datasets.upload"
    #             raise Exception(msg)
    #
    #     if ontology is not None:
    #         if not issubclass(type(ontology), Ontology):
    #             msg = "Required Ontology, not {}".format(type(ontology))
    #             raise Exception(msg)
    #
    #         if not ontology.stored:
    #             msg = "{} must be stored on the server, use <Serene>.ontologies.upload"
    #             raise Exception(msg)
    #
    #     self._name = name if name is not None else gen_id()
    #     self._dataset = dataset
    #     self._ontology = ontology
    #     self._VERSION = "0.1"
    #     self._id = None
    #     self._stored = False  # is stored on the server?
    #     self._date_created = None
    #     self._date_modified = None
    #
    #     # semantic model
    #     self._semantic_model = BaseSemantic()
    #
    #     # the mapping object from Column -> Mapping
    #     # this should always contain the complete Column set,
    #     # so that the user can easily see what is and isn't
    #     # yet mapped.
    #     self._mapping = {}
    #     if self._dataset is not None:
    #         for i, column in enumerate(self._dataset.columns):
    #             self._mapping[column] = Mapping(
    #                 column,
    #                 node=None)
    #
    # def update(self, blob, dataset_endpoint, ontology_endpoint):
    #     """
    #     Create the object from json directly
    #
    #     :param blob:
    #     :param dataset_endpoint:
    #     :param ontology_endpoint:
    #     :return:
    #     """
    #     if 'id' in blob:
    #         self._stored = True
    #         self._date_created = convert_datetime(blob['dateCreated'])
    #         self._date_modified = convert_datetime(blob['dateModified'])
    #         self._id = int(blob['id'])
    #
    #     if 'name' in blob:
    #         self._name = blob['name']
    #     reader = SSDReader(blob, dataset_endpoint, ontology_endpoint)
    #
    #     self._ontology = reader.ontology
    #     self._dataset = reader.dataset
    #     self._semantic_model = reader.semantic_model
    #
    #     # build up the mapping through the interface...
    #     for col, ds in reader.mapping:
    #         self.map(col, ds, _init=False)
    #
    #     self._semantic_model.summary()
    #
    #     return self
    #
    # def _find_column(self, column):
    #     """
    #     Searches for a Column in the system given a abbreviated
    #     Column type. e.g. Column("name")
    #
    #     would return the full Column object (if it exists in the system):
    #
    #     Column("name", index=1, file="something.csv")
    #
    #     :param column: An abbreviated Column type
    #     :return: The actual Column in the system, or an error if not found or ambiguous
    #     """
    #     col = Column.search(self._dataset.columns, column)
    #     if col is None:
    #         msg = "Failed to find column: {}".format(column)
    #         _logger.error(msg)
    #         raise Exception(msg)
    #     return col
    #
    # def _find_data_node(self, data_node):
    #     """
    #     Searches for a Datanode in the system given a abbreviated
    #     DataNode type. e.g. DataNode("name")
    #
    #     would return the full DataNode object (if it exists in the system):
    #
    #     DataNode("name", parent=ClassNode("Person"))
    #
    #     :param data_node: An abbreviated DataNode type
    #     :return: The actual DataNode in the system, or an error if not found or ambiguous
    #     """
    #     data_nodes = self._ontology.idata_nodes
    #     dn = DataProperty.search(data_nodes, data_node)
    #     if dn is None:
    #         msg = "Failed to find DataNode: {}".format(data_node)
    #         _logger.error(msg)
    #         raise Exception(msg)
    #     return dn
    #
    # def _find_class(self, cls):
    #     """
    #     Searches for a ClassNode in the system given a abbreviated
    #     ClassNode type. e.g. ClassNode('Person')
    #
    #     would return the full Link object (if it exists in the system):
    #
    #     ClassNode("Person", ["name", "dob", "phone-number"])
    #
    #     :param cls: An abbreviated ClassNode type
    #     :return: The actual ClassNode in the system, or an error if not found or ambiguous
    #     """
    #     classes = set(self._ontology.iclass_nodes)
    #
    #     c = Class.search(classes, cls)
    #     if c is None:
    #         msg = "Failed to find Class: {}".format(c)
    #         _logger.error(msg)
    #         raise Exception(msg)
    #     return c
    #
    # def _find_link(self, link):
    #     """
    #     Searches for a Link in the system given a abbreviated
    #     Link type. e.g. Link('owns')
    #
    #     would return the full Link object (if it exists in the system):
    #
    #     Link(src=ClassNode("Person"), dst=ClassNode("Car"), relationship='owns')
    #
    #     :param link: An abbreviated Link type
    #     :return: The actual Link in the system, or an error if not found or ambiguous
    #     """
    #     link_ = ObjectProperty.search(self.links, link)
    #     if link_ is None:
    #         msg = "Failed to find Link: {}".format(link)
    #         _logger.error(msg)
    #         raise Exception(msg)
    #     return link_
    #
    # def find(self, item):
    #     """
    #     Helper function to locate objects using shorthands e.g.
    #
    #     ssd.find(Column("name"))
    #
    #     :param item: Column, ClassNode, Link or DataNode object
    #     :return:
    #     """
    #     if issubclass(type(item), DataProperty):
    #         return self._find_data_node(item)
    #     elif issubclass(type(item), Column):
    #         return self._find_column(item)
    #     elif issubclass(type(item), Class):
    #         return self._find_class(item)
    #     elif issubclass(type(item), ObjectProperty):
    #         return self._find_link(item)
    #     else:
    #         raise TypeError("This type is not supported in find().")
    #
    # def map(self, column, node, _init=True):
    #     """
    #     Adds a link between the column and data_node for the
    #     mapping.
    #
    #     :param column: The source Column
    #     :param data_node: The destination DataNode
    #     :param _init: if the mapping comes from the SSDreader, then we should not change nodes
    #     :return:
    #     """
    #     if issubclass(type(column), str):
    #         column = Column(column)
    #
    #     if issubclass(type(node), str):
    #         if '.' in node:
    #             node = DataProperty(*node.split('.'))
    #         else:
    #             # TODO: ClassNode should be supported for foreign keys!
    #             node = Class(node)
    #
    #     return self._map(column, node, predicted=False, _init=_init)
    #
    # def _map(self, column, node, predicted=False, _init=True):
    #     """
    #     [Internal] Adds a link between the column and data_node for the
    #     mapping.
    #
    #     :param column: The source Column
    #     :param node: The destination DataNode
    #     :param predicted: Is the mapping predicted or not (internal)
    #     :param _init: if the mapping comes from the SSDreader, then we should not change nodes or semantic model
    #     :return:
    #     """
    #     assert type(column) == Column
    #     assert type(node) == DataProperty
    #
    #     col = self._find_column(column)
    #     if _init:
    #         dn = self._find_data_node(node)
    #     else:
    #         dn = node
    #
    #     # add the mapping element to the table...
    #     self._mapping[col] = Mapping(col, dn, None, predicted)
    #
    #     # grab the parent from the ontology
    #     if _init:
    #         parent = Class.search(self._ontology.iclass_nodes, dn.parent)
    #     else:
    #         parent = dn.parent
    #     if parent is None:
    #         msg = "{} does not exist in the ontology.".format(parent)
    #         raise Exception(msg)
    #
    #     # by making this mapping, the class node is now in the SemanticModel...
    #     if _init:
    #         self._semantic_model.add_class_node(parent, add_data_nodes=False)
    #
    #     target = ObjectProperty(dn.name, parent, dn)
    #     link = ObjectProperty.search(self._ontology.ilinks, target)
    #     if link is None:
    #         msg = "{} does not exist in the ontology.".format(target)
    #         raise Exception(msg)
    #
    #     self._semantic_model.add_link(target)
    #
    #     return self
    #
    # def link(self, src, dst, relationship):
    #     """
    #     Adds a link between ClassNodes. The relationship must
    #     exist in the ontology
    #
    #     :param src: Source ClassNode
    #     :param dst: Destination ClassNode
    #     :param relationship: The label for the link between the src and dst nodes
    #     :return: Updated SSD
    #     """
    #     if issubclass(type(src), str):
    #         src = Class(src)
    #
    #     if issubclass(type(dst), str):
    #         dst = Class(dst)
    #
    #     s_class = self._find_class(src)
    #     d_class = self._find_class(dst)
    #
    #     # now check that the link is in the ontology...
    #     parent_links = self._ontology.ilinks
    #     target_link = ObjectProperty(relationship, s_class, d_class)
    #     link = ObjectProperty.search(parent_links, target_link)
    #
    #     if link is None:
    #         msg = "Link {} does not exist in the ontology.".format(relationship)
    #         _logger.error(msg)
    #         raise Exception(msg)
    #     elif link in self.links:
    #         msg = "Link {} is already in the links".format(link)
    #         _logger.info(msg)
    #     else:
    #         target_link.prefix = link.prefix
    #
    #         # if it is ok, then add the new link to the SemanticModel...
    #         self._semantic_model.add_link(target_link)
    #
    #     return self
    #
    # def sample(self, column, n=10):
    #     """
    #     Samples the column
    #
    #     ssd.sample(Column('name'))
    #
    #     :param column: The column object to sample
    #     :param n: The number of samples to take
    #     :return: List of samples from the column
    #     """
    #     # first we check that this request is valid...
    #     col = self._find_column(column)
    #
    #     # next we pull a sample from the Column dataframe...
    #     samples = col.df[col.name].sample(n).values
    #
    #     return list(samples)
    #
    # def remove(self, item):
    #     """
    #     Remove the item. The item can be a
    #     a dataproperty, column or link.
    #
    #     :return:
    #     """
    #     if type(item) == DataProperty:
    #         elem = self._find_data_node(item)
    #         key = None
    #         value = None
    #         for k, v in self._mapping.items():
    #             if v.node == elem:
    #                 key = k
    #                 value = v
    #                 break
    #
    #         # we put the default mapping back in...
    #         self._mapping[key] = Mapping(
    #             value.column,
    #             node=None
    #         )
    #         # note that we now need to check whether
    #         # we should remove the classNode from the
    #         # SemanticModel
    #         old = set(self._semantic_model.class_nodes)
    #         new = set(self.class_nodes)
    #         targets = new - old
    #         for t in targets:
    #             self._semantic_model.remove_node(t)
    #
    #     elif type(item) == Column:
    #         elem = self._find_column(item)
    #         key = None
    #         for k, v in self._mapping.items():
    #             if v.col == elem:
    #                 key = k
    #                 break
    #         del self._mapping[key]
    #
    #     elif type(item) == ObjectProperty:
    #         elem = self._find_link(item)
    #
    #         self._semantic_model.remove_link(elem)
    #     else:
    #         raise TypeError("This type is not supported in remove().")
    #
    #     return self
    #
    # def show(self):
    #     """
    #     Shows the Semantic Modeller in a png. This uses the SSDVisualizer helper
    #     to launch the graphviz image.
    #
    #     :return: None
    #     """
    #     self.summary()
    #     SSDVisualizer(self).show()
    #     return
    #
    # def summary(self):
    #     """Prints out the summary string..."""
    #     print()
    #     print("{}:".format(self._name))
    #     print(self._full_str())
    #
    # @property
    # def name(self):
    #     """Returns the current name of the SSD"""
    #     return self._name
    #
    # @property
    # def id(self):
    #     """Returns the current ID of the SSD"""
    #     return self._id
    #
    # @property
    # def stored(self):
    #     """The stored flag if the ssd is on the server"""
    #     return self._stored
    #
    # @property
    # def version(self):
    #     """The SSD version"""
    #     return self._VERSION
    #
    # @property
    # def ontology(self):
    #     """The read-only list of ontology objects"""
    #     return self._ontology
    #
    # @property
    # def ssd(self):
    #     """The SSD as a Python dictionary"""
    #     builder = SSDJsonWriter(self)
    #     return builder.to_dict()
    #
    # @property
    # def json(self):
    #     """The SSD as a JSON string"""
    #     builder = SSDJsonWriter(self)
    #     return builder.to_json()
    #
    # @property
    # def dataset(self):
    #     """returns the dataset provided"""
    #     return self._dataset
    #
    # @property
    # def model(self):
    #     """The read-only semantic model"""
    #     return self._semantic_model
    #
    # @property
    # def predictions(self):
    #     """The list of all predicted mappings"""
    #     return list(m for m in self.mappings if m.predicted)
    #
    # @property
    # def mappings(self):
    #     """The list of all current mappings"""
    #     return list(self._mapping.values())
    #
    # @property
    # def columns(self):
    #     """The list of columns currently used"""
    #     return list(self._mapping.keys())
    #
    # @property
    # def data_nodes(self):
    #     """All available data nodes"""
    #     return list(m.node for m in self.mappings)
    #
    # @property
    # def class_nodes(self):
    #     """All available class nodes"""
    #     return self._semantic_model.class_nodes
    #
    # @property
    # def data_links(self):
    #     """ClassNode - DataNode links in the semantic model"""
    #     # # grab the object links from the model...
    #     return self._semantic_model.data_links
    #
    # @property
    # def class_links(self):
    #     """ClassNode - ClassNode links in the semantic model"""
    #     # # grab the object links from the model...
    #     return [link for link in self._semantic_model.links
    #             if link.link_type == ObjectProperty.OBJECT_LINK]
    #
    # @property
    # def links(self):
    #     """
    #         The links in the SSD. The links come from 2 sources. The object links come from
    #         what's defined in the semantic model. The data links come
    #         from the relevant mappings to the columns. The other links
    #         are not necessary.
    #     """
    #     return self.data_links + self.class_links
    #
    # def _full_str(self):
    #     """
    #     Displays the maps of columns to the data nodes of the Ontology
    #     :return:
    #     """
    #     map_str = [str(m) for m in self.mappings]
    #     link_str = [str(link) for link in self.links if link.link_type == ObjectProperty.OBJECT_LINK]
    #
    #     items = map_str + link_str
    #     full_str = '\n\t'.join(items)
    #
    #     return "[\n\t{}\n]".format(full_str)
    #
    # def __repr__(self):
    #     return "SSD({}, {})".format(self._name, self._dataset.filename)


class SSDReader(object):
    pass
    # """
    # The SSDReader is a helper object used to parse an SSD json
    # blob from the server.
    # """
    # def __init__(self, blob, dataset_endpoint, ontology_endpoint):
    #     """Builds up the relevant properties from the json blob `json`
    #         note that we need references to the endpoints to ensure
    #         that the information is up-to-date with the server.
    #     """
    #     self._ds_endpoint = dataset_endpoint
    #     self._on_endpoint = ontology_endpoint
    #     self._ontology = self._find_ontology(blob)
    #     self._dataset = self._find_dataset(blob)
    #     self._semantic_model = self._build_semantic(blob, self._ontology)
    #     self._mapping = self._build_mapping(blob)
    #
    # @property
    # def ontology(self):
    #     return self._ontology
    #
    # @property
    # def dataset(self):
    #     return self._dataset
    #
    # @property
    # def semantic_model(self):
    #     return self._semantic_model
    #
    # @property
    # def mapping(self):
    #     return self._mapping
    #
    # def _find_ontology(self, json):
    #     """Pulls the ontology reference from the SSD and queries the server"""
    #     ontologies = json['ontology']
    #     # TODO: Only one ontology used!! Should be a sequence!
    #     return self._on_endpoint.get(ontologies[0])
    #
    # def _find_dataset(self, json):
    #     """Attempts to grab the dataset out from the json string"""
    #
    #     columns = [c['attribute'] for c in json['mappings']]
    #
    #     if not len(columns):
    #         msg = "No columns present in ssd file mappings."
    #         raise Exception(msg)
    #
    #     col_map = self._ds_endpoint.columns
    #
    #     if columns[0] not in col_map:
    #         msg = "Column {} does not appear on the server".format(columns[0])
    #         raise Exception(msg)
    #
    #     ds_key = col_map[columns[0]].datasetID
    #
    #     return self._ds_endpoint.get(ds_key)
    #
    # @staticmethod
    # def _get_data_nodes(semantic_json):
    #     """
    #     Extract the datanodes from the json blob
    #     :param semantic_json:
    #     :return:
    #     """
    #
    #     class_table = {n["id"]: Class(name=n["label"], prefix=n["prefix"], idx=n["id"])
    #                    for n in semantic_json["nodes"]
    #                    if n["type"] == "ClassNode"}
    #
    #     data_links = [(n["source"], n["target"], n["label"], n["prefix"])
    #                   for n in semantic_json["links"]
    #                   if n["type"] == ObjectProperty.DATA_LINK]
    #
    #     # fill out lookups for data node ids
    #     lookup = {}
    #     for link in data_links:
    #         src, dst, name, prefix = link
    #         class_node = class_table[src]
    #         lookup[dst] = DataProperty(class_node, name, prefix=prefix)
    #
    #     return lookup
    #
    # def _build_mapping(self, json):
    #     """Builds the mapping from the json string"""
    #     jsm = json["semanticModel"]
    #     raw_map = json["mappings"]
    #
    #     raw_data_nodes = self._get_data_nodes(jsm)
    #
    #     data_nodes = {dn_id: DataProperty.search(self._semantic_model.data_nodes, dn)
    #                   for (dn_id, dn) in raw_data_nodes.items()}
    #
    #     column_map = {col.id: col for col in self._dataset.columns}
    #
    #     values = []
    #     for link in raw_map:
    #         cid = column_map[link["attribute"]]
    #         node = data_nodes[link["node"]]
    #         values.append((cid, node))
    #
    #     return values
    #
    # @staticmethod
    # def _build_semantic(blob, ontology):
    #     """Builds the semantic model from a json string"""
    #     sm = BaseSemantic()
    #     jsm = blob["semanticModel"]
    #     raw_nodes = jsm["nodes"]
    #     links = jsm["links"]
    #
    #     class_table = {n["id"]: Class(n["label"], prefix=n["prefix"], idx=n["id"]) for n in raw_nodes
    #                    if n["type"] == "ClassNode"}
    #
    #     # print(">>>> class_table >>>>>", class_table)
    #
    #     class_links = [(n["source"], n["target"], n["label"], n["prefix"])
    #                    for n in links
    #                    if n["type"] == ObjectProperty.OBJECT_LINK]
    #
    #     # print("===== class_links ====")
    #     # for c in class_links:
    #     #     print(c)
    #
    #     data_links = [(n["source"], n["target"], n["label"], n["prefix"])
    #                   for n in links
    #                   if n["type"] == ObjectProperty.DATA_LINK]
    #
    #     # print("+++++ data_links +++++", data_links)
    #
    #     # fill out lookups for data properties
    #     lookup = defaultdict(list)
    #     for link in data_links:
    #         src, dst, name, prefix = link
    #         lookup[src].append((dst, name, prefix))
    #
    #     # print("lookup: ", lookup)
    #
    #     # Note: we can have class nodes which do not have any data nodes
    #     class_node_lookup = {}
    #     for cls in class_table:
    #         class_node = class_table[cls]
    #         cn = Class.search(ontology.iclass_nodes, class_node)
    #         # now we change the data nodes on this class node to the ones which exist within the semantic model
    #         data_nodes = lookup[cls]
    #         cn = Class(cn.name, nodes=None, prefix=cn.prefix)
    #         if len(data_nodes):
    #             for (dst, name, prefix) in data_nodes:
    #                 dns = DataProperty.search(ontology.idata_nodes, DataProperty(cn, name, prefix=prefix))
    #                 cn.nodes.append(dns)
    #         # now add this class node together with data nodes
    #         # data property links will be automatically added
    #         sm.add_class_node(cn)
    #         class_node_lookup[cls] = cn  # update class table with the correct class node
    #
    #     # finally we add the class links
    #     for link in class_links:
    #         src, dst, name, prefix = link
    #         # print("Processing LINK {}, {}, {}, {}".format( src, name, dst, prefix))
    #         # source class node...
    #         scn = class_node_lookup[src]
    #         # dst class node...
    #         dcn = class_node_lookup[dst]
    #         # print("ADDING LINK {}, {}, {}, {}".format(scn, name, dcn, prefix))
    #         sm.link(scn.name, name, dcn.name, prefix=prefix)
    #
    #     # print("~~~class nodes: ", sm.class_nodes)
    #
    #     return sm


class SSDJsonWriter(object):
    pass
    # """
    # Helper class to build up the json output for the SSD file.
    # """
    # def __init__(self, ssd):
    #     """
    #
    #     :param ssd:
    #     """
    #     self._ssd = ssd
    #
    #     self._all_nodes = [n for n in
    #                        self._ssd.class_nodes + self._ssd.data_nodes
    #                        if n is not None]
    #
    #     self._node_map = {m: i for i, m in enumerate(self._all_nodes)}
    #     self._attr_map = {m: i for i, m in enumerate(self._ssd.mappings)}
    #
    # def to_dict(self):
    #     """Builds the dictionary representation of the SSD"""
    #     d = OrderedDict()
    #     d["name"] = self._ssd.name
    #     d["ontologies"] = [self._ssd.ontology.id]
    #     d["semanticModel"] = self.semantic_model
    #     d["mappings"] = self.mappings
    #     return d
    #
    # def to_json(self):
    #     """Builds the complete json object string"""
    #     return json.dumps(self.to_dict())
    #
    # @property
    # def semantic_model(self):
    #     """Builds out the .ssd semantic model section..."""
    #
    #     return {
    #         "nodes": [node.ssd_output(index)
    #                   for index, node in enumerate(self._all_nodes)],
    #         "links": [link.ssd_output(index, self._node_map)
    #                   for index, link in enumerate(self._ssd.links)]
    #     }
    #
    # @property
    # def mappings(self):
    #     """BUilds out the .ssd mapping section.."""
    #     return [
    #         {
    #             "attribute": m.column.id,
    #             "node": self._node_map[m.node]
    #         } for m in self._ssd.mappings if m.node is not None]