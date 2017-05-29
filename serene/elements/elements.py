"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the raw elements object
"""
import collections
import logging
import types

from serene.utils import Searchable


# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.WARN)


class Column(Searchable):
    """
    Holds a reference to the original data column
    """
    # now we setup the search parameters...
    getters = [
        lambda col: col.name,
        lambda col: col.index,
        lambda col: col.filename
    ]

    def __init__(self, name, df=None, filename=None, index=None):
        """

        :param name:
        :param df:
        :param filename:
        :param index:
        """
        self.id = None
        self.size = None
        self.datasetID = None
        self.dataset = None
        self.sample = None
        self.logicalType = None
        self.df = df
        self.filename = filename
        self.name = name
        self.index = index
        super().__init__()

    def update(self, json, dataset):
        """
        Update a Column object using a json table...

        :param json: JSON table from the server
        """
        self.index = json['index']
        self.filename = json['path']
        self.name = json['name']
        self.id = json['id']
        self.size = json['size']
        self.datasetID = json['datasetID']
        self.sample = json['sample']
        self.logicalType = json['logicalType']

        self.dataset = dataset

        return self

    def __repr__(self):
        """
        Displays the column and the name
        :return:
        """
        return "Column({})".format(self.name)

    def __eq__(self, other):
        return (
            (other.name == self.name) and
            (other.id == self.id) and
            (other.datasetID == self.datasetID)
        )

    def __hash__(self):
        return hash((self.name, self.id, self.datasetID))


class Mapping(object):
    """
        A Mapping object that describes the link between a column and
        a data node.
    """
    def __init__(self, column, node=None, predicted=False):
        if type(column) != Column:
            raise TypeError("Column type required for 'column' in Mapping object")

        if (node is not None) and (type(node) != DataProperty):
            raise TypeError("DataNode type required for 'node' in Mapping object")

        self.column = column
        self.node = node
        self.predicted = predicted

    def __repr__(self):
        s = "{} -> {}".format(self.column, self.node)

        # we add an asterix to denote predicted values...
        if self.predicted:
            s = '{} [*]'.format(s)
        return s


class Class(Searchable):
    """
        ClassNode objects hold the 'types' of the ontology or semantic model.
        A ClassNode can have multiple DataNodes. A DataNode corresponds to an
        attribute or a column in a dataset.
        A ClassNode can link to another ClassNode via a Link.
    """
    # the search parameters...
    getters = [
        lambda node: node.label,
        lambda node: node.prefix if node.prefix else None,
        lambda node: node.parent if node.parent else None
    ]

    def __init__(self, label, nodes=None, prefix=None, parent=None, idx=None):
        """
        A ClassNode is initialized with a name, a list of string nodes
        and optional prefix and/or a parent ClassNode

        :param label: The string name of the ClassNode
        :param nodes: A list of strings to initialze the DataNode objects
        :param prefix: The URI prefix
        :param parent: The parent object if applicable
        :param idx: node id to distinguish between instances of the same class
        """
        self.label = label
        self.prefix = prefix
        self.parent = parent
        self.idx = idx

        if nodes is None:
            self.nodes = []
        elif issubclass(type(nodes), list) or issubclass(type(nodes), types.GeneratorType):
            self.nodes = [DataProperty(self, n, dtype=str, prefix=prefix) for n in nodes]
        else:
            self.nodes = [DataProperty(self, n, dtype=dtype, prefix=prefix) for n, dtype in nodes.items()]

    def ssd_output(self, ident):
        """
        The output function has not
        :return:
        """
        if self.prefix is None:
            msg = "There is no prefix specified for {}".format(self)
            raise Exception(msg)
        if self.idx is not None:
            ident = self.idx
        return {
            "id": ident,
            "label": self.label,
            "prefix": self.prefix,
            "type": "Class"
        }

    def __repr__(self):
        nodes = [n.label for n in self.nodes]

        return "Class({}, [{}])".format(self.label, ", ".join(nodes))

    def __eq__(self, other):
        return (self.label == other.label) \
               and (self.prefix == other.prefix)

    def __hash__(self):
        return hash((self.label, self.prefix))


class DataProperty(Searchable):
    """
        A DataNode is an attribute of a ClassNode. This can correspond to a
        column in a dataset.
    """
    # the search parameters...
    getters = [
        lambda node: node.label,
        lambda node: node.parent.label if node.parent else None,
        lambda node: node.parent.prefix if node.parent else None
    ]

    def __init__(self, *names, dtype=str, prefix=None):
        """
        A DataProperty is initialized with name and a parent Class object.
        A DataProperty can be initialized in the following ways:

        DataProperty(ClassNode("Person"), "name")
        DataProperty("Person", "name)
        DataProperty("name")

        :param names: The name of the parent class and the name of the DataProperty
        """
        self.dtype = dtype
        self.prefix = prefix

        if len(names) == 1:
            # initialized with DataProperty("name") - for lookups only...
            self.label = names[0]
            self.parent = None

        elif len(names) == 2:
            # here the first element is now the parent...
            parent = names[0]

            if type(parent) == Class:
                # initialized with DataProperty(Class("Person"), "name")
                self.parent = parent
            else:
                # initialized with DataProperty("Person", "name")
                self.parent = Class(parent)
            self.label = names[1]

        else:
            msg = "Insufficient args for DataNode construction."
            raise Exception(msg)

        super().__init__()

    def ssd_output(self, ident):
        """
        The output function has not
        :return:
        """
        if self.parent is None:
            label = self.label
        else:
            label = "{}.{}".format(self.parent.label, self.label)

        return {
            "id": ident,
            "label": label,
            "type": "DataProperty"
        }

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __eq__(self, other):
        if type(other) is type(self):
            if self.parent is not None and other.parent is not None:
                return self.label == other.label and \
                       self.parent.label == other.parent.label
            else:
                return self.label == other.label
        return False

    def __repr__(self):
        if self.parent:
            return "DataProperty({}, {})".format(self.parent.label, self.label)
        else:
            return "DataProperty({})".format(self.label)

    def __hash__(self):
        if self.parent:
            return hash((self.parent.label, self.label))
        else:
            return hash(self.label)


class ObjectProperty(Searchable):
    """
        A Link is a relationship between Class->Class or Class->DataProperty.
    """
    @staticmethod
    def node_match(node):
        if node.src is None:
            return None
        if node.dst is None:
            return None
        return node.src.label, node.dst.label

    # the search parameters...
    getters = [
        lambda node: node.label,
        lambda node: ObjectProperty.node_match(node),
        lambda node: node.prefix
    ]

    # special link names...
    SUBCLASS = "subclass"
    OBJECT_LINK = "ObjectPropertyLink"
    DATA_LINK = "DataPropertyLink"

    def __init__(self, label, src=None, dst=None, prefix=None):
        """
        The link can be initialized with a name, and also
        holds the source and destination ClassNode references.

        :param label: The link name
        :param src: The source ClassNode
        :param dst: The destination ClassNode
        """
        self.label = label
        self.src = src
        self.dst = dst
        self.prefix = prefix
        if type(self.dst) == Class:
            self.link_type = self.OBJECT_LINK
        else:
            self.link_type = self.DATA_LINK

    def ssd_output(self, index, index_map):
        """
        Returns the SSD output for a link. Note that we need
        the local reference to the index map, so that we can
        put the correct source/target link IDs.

        :return: Dictionary with the SSD link labels
        """
        if issubclass(type(self.dst), Class) and self.prefix is None:
            msg = "{} has no prefix namespace specified.".format(self)
            raise ValueError(msg)

        output = {
            "id": index,
            "source": index_map[self.src],
            "target": index_map[self.dst],
            "label": self.label,
            "type": self.link_type,
            "prefix": self.prefix
        }
        return {k:v for k, v in output.items() if v is not None}

    def __repr__(self):
        if (self.src is None) or (self.dst is None):
            return "Link({})".format(self.label)
        elif self.link_type == self.OBJECT_LINK:
            return "ClassNode({}) -> Link({}) -> ClassNode({})" \
                .format(self.src.label, self.label, self.dst.label)
        else:
            return "ClassNode({}) -> Link({}) -> DataNode({})" \
                .format(self.src.label, self.label, self.dst.label)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.label == other.label) and \
               (self.src == other.src) and \
               (self.dst == other.dst)

    def __hash__(self):
        return hash((self.src, self.dst, self.label))


class ObjectPropertyList(collections.MutableSequence):
    """
    Container type for Link objects in the Semantic Source Description
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not isinstance(v, ObjectProperty):
            raise TypeError("Only ObjectProperty types permitted")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        return '\n'.join(str(link) for link in self.list)


class SSDSearchable(Searchable):
    pass


class ClassNode(SSDSearchable):

    # the search parameters...
    getters = [
        lambda node: node.label,
        lambda node: node.index,  # if node.index is not None else None,
        lambda node: node.prefix if node.prefix else None
    ]

    def __init__(self, label, index=None, prefix=None):
        self._label = label
        self._index = index
        self._prefix = prefix
        self._type = "ClassNode"

        super().__init__()

    @property
    def full_label(self):
        return self.label

    @property
    def label(self):
        return self._label

    @property
    def index(self):
        return self._index

    @property
    def prefix(self):
        return self._prefix

    @property
    def type(self):
        return self._type

    def __repr__(self):
        if self.index is None:
            return "ClassNode({}, {})".format(self.label, self.prefix)
        else:
            return "ClassNode({}, {}, {})".format(self.label, self.prefix, self.index)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (type(self) == type(other)) \
               and (self.label == other.label) \
               and (str(self.prefix) == str(other.prefix)) \
               and (self.index == other.index)

    def __hash__(self):
        return hash((self.index, self.label, str(self.prefix), type(self)))


class DataNode(SSDSearchable):

    # the search parameters...
    getters = [
        lambda node: node.full_label,
        lambda node: node.index,
        lambda node: node.prefix if node.prefix else None,
        lambda node: node.class_node if node.class_node else None
    ]

    def __init__(self, class_node, label, index=None, prefix=None, dtype=str):
        """
        A DataNode is initialized with name and a parent Class object.
        A DataNode can be initialized in the following ways:

        DataNode(ClassNode("Person"), "name")
        DataNode("Person", "name)

        :param class_node: The class node parent
        :param label: The name of the parent class and the name of the DataNode
        """
        self._dtype = dtype
        self._prefix = prefix

        if type(class_node) != ClassNode:
            msg = "DataNode must be initialized with a ClassNode"
            raise ValueError(msg)

        self._class_node = class_node
        self._label = label
        self._type = "DataNode"
        self._index = index

        super().__init__()

    @property
    def label(self):
        return self._label

    @property
    def class_node(self):
        return self._class_node

    @property
    def index(self):
        return self._index

    @property
    def prefix(self):
        return self._prefix

    @property
    def type(self):
        return self._type

    @property
    def dtype(self):
        return self._dtype

    @property
    def full_label(self):
        return self.class_node.label + '.' + self._label

    def __repr__(self):
        if self.index is not None:
            return "DataNode({}, {}, {})".format(self.class_node, self.label, self.index)
        else:
            return "DataNode({}, {})".format(self.class_node, self.label)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (type(self) == type(other)) \
               and (self.label == other.label) \
               and (str(self.prefix) == str(other.prefix)) \
               and (self.index == other.index) \
               and (self.class_node == other.class_node)

    def __hash__(self):
        return hash((self.label, str(self.prefix), self.class_node, self.index))


class SSDLink(object):
    """
    Link objects for the SSD semantic model
    """
    def __init__(self, label, prefix):
        self.label = label
        self.prefix = prefix
        self.type = None

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.label == other.label) \
               and (str(self.prefix) == str(other.prefix))

    def __hash__(self):
        return hash((self.label, str(self.prefix)))


class DataLink(SSDLink):
    """
    Link between ClassNode -> DataNode
    """
    def __init__(self, label, prefix=None):
        super().__init__(label, prefix)
        self.type = "DataPropertyLink"

    def __repr__(self):
        return "DataLink({})".format(self.label)


class ObjectLink(SSDLink):
    """
    Link between ClassNode -> ClassNode
    """
    def __init__(self, label, prefix):
        super().__init__(label, prefix)
        self.type = "ObjectPropertyLink"

    def __repr__(self):
        return "ObjectLink({})".format(self.label)


class ColumnLink(SSDLink):
    """
    Link between DataNode -> Column
    """
    def __init__(self, label, prefix=None):
        super().__init__(label, prefix)
        self.type = "ColumnLink"

    def __repr__(self):
        return "ColumnLink({}, {})".format(self.label, self.prefix)

    def __eq__(self, other):
        return (self.label == other.label) and (self.type == other.type)


class ClassInstanceLink(DataLink):
    """
    Link between ClassNode -> DataNode
    """
    def __init__(self, label, prefix):
        super().__init__(label, prefix)
        self.type = "ClassInstanceLink"

    def __repr__(self):
        return "ClassLink({})".format(self.label)


class SubClassLink(SSDLink):
    """
    Link between ClassNode -> ClassNode of type subclass
    """
    def __init__(self, label, prefix):
        super().__init__(label, prefix)
        self.type = "SubClassLink"

    def __repr__(self):
        return "SubClassLink({})".format(self.label)
