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

    def update(self, json):
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
        lambda node: node.name,
        lambda node: node.prefix if node.prefix else None,
        lambda node: node.parent if node.parent else None
    ]

    def __init__(self, name, nodes=None, prefix=None, parent=None, idx=None):
        """
        A ClassNode is initialized with a name, a list of string nodes
        and optional prefix and/or a parent ClassNode

        :param name: The string name of the ClassNode
        :param nodes: A list of strings to initialze the DataNode objects
        :param prefix: The URI prefix
        :param parent: The parent object if applicable
        :param idx: node id to distinguish between instances of the same class
        """
        self.name = name
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
            "label": self.name,
            "prefix": self.prefix,
            "type": "ClassNode"
        }

    def __repr__(self):
        nodes = [n.name for n in self.nodes]

        if self.parent is None:
            parent = ""
        else:
            parent = ", parent={}".format(self.parent.name)

        #return "ClassNode({}, [{}]{})".format(self.name, ", ".join(nodes), parent)
        return "ClassNode({})".format(self.name)

    def __eq__(self, other):
        return (self.name == other.name) \
               and (self.prefix == other.prefix) # \
               #and (self.nodes == other.nodes)

    def __hash__(self):
        return hash((self.name, self.prefix)) #, frozenset(self.nodes)))


class DataProperty(Searchable):
    """
        A DataNode is an attribute of a ClassNode. This can correspond to a
        column in a dataset.
    """
    # the search parameters...
    getters = [
        lambda node: node.name,
        lambda node: node.parent.name if node.parent else None,
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
            self.name = names[0]
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
            self.name = names[1]

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
            label = self.name
        else:
            label = "{}.{}".format(self.parent.name, self.name)

        return {
            "id": ident,
            "label": label,
            "type": "DataNode"
        }

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __eq__(self, other):
        if type(other) is type(self):
            if self.parent is not None and other.parent is not None:
                return self.name == other.name and \
                    self.parent.name == other.parent.name
            else:
                return self.name == other.name
        return False

    def __repr__(self):
        if self.parent:
            return "DataProperty({}, {})".format(self.parent.name, self.name)
        else:
            return "DataProperty({})".format(self.name)

    def __hash__(self):
        if self.parent:
            return hash((self.parent.name, self.name))
        else:
            return hash(self.name)


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
        return node.src.name, node.dst.name

    # the search parameters...
    getters = [
        lambda node: node.name,
        lambda node: ObjectProperty.node_match(node),
        lambda node: node.prefix
    ]

    # special link names...
    SUBCLASS = "subclass"
    OBJECT_LINK = "ObjectPropertyLink"
    DATA_LINK = "DataPropertyLink"

    def __init__(self, name, src=None, dst=None, prefix=None):
        """
        The link can be initialized with a name, and also
        holds the source and destination ClassNode references.

        :param name: The link name
        :param src: The source ClassNode
        :param dst: The destination ClassNode
        """
        self.name = name
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
            "label": self.name,
            "type": self.link_type,
            "prefix": self.prefix
        }
        return {k:v for k, v in output.items() if v is not None}

    def __repr__(self):
        if (self.src is None) or (self.dst is None):
            return "Link({})".format(self.name)
        elif self.link_type == self.OBJECT_LINK:
            return "ClassNode({}) -> Link({}) -> ClassNode({})" \
                .format(self.src.name, self.name, self.dst.name)
        else:
            return "ClassNode({}) -> Link({}) -> DataNode({})" \
                .format(self.src.name, self.name, self.dst.name)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.name == other.name) and \
               (self.src == other.src) and \
               (self.dst == other.dst)

    def __hash__(self):
        return hash((self.src, self.dst, self.name))


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
        lambda node: node.index if node.index else None,
        lambda node: node.prefix if node.prefix else None
    ]

    def __init__(self, label, index=None, prefix=None):
        self.label = label
        self.index = index
        self.prefix = prefix
        self.type = "ClassNode"

        super().__init__()

    def __repr__(self):
        if self.index is None:
            return "ClassNode({}, {})".format(self.label, self.prefix)
        else:
            return "ClassNode({}, {}, {})".format(self.label, self.prefix, self.index)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.label == other.label) \
               and (self.prefix == other.prefix)

    def __hash__(self):
        return hash((self.label, self.prefix))


class DataNode(SSDSearchable):

    # the search parameters...
    getters = [
        lambda node: node.full_label,
        lambda node: node.prefix if node.prefix else None
    ]

    def __init__(self, *labels, dtype=str, prefix=None):
        """
        A DataNode is initialized with name and a parent Class object.
        A DataNode can be initialized in the following ways:

        DataNode(ClassNode("Person"), "name")
        DataNode("Person", "name)
        DataNode("name")

        :param labels: The name of the parent class and the name of the DataNode
        """
        self.dtype = dtype
        self.prefix = prefix

        if len(labels) == 1:
            # initialized with DataNode("name") - for lookups only...
            self._label = labels[0]

        elif len(labels) == 2:
            # here the first element is now the parent...
            class_node = labels[0]

            if type(class_node) == ClassNode:
                # initialized with DataNode(Class("Person"), "name")
                self.class_node = class_node
            else:
                # initialized with DataNode("Person", "name")
                self.class_node = ClassNode(class_node)
            self._label = labels[1]
            self.type = "DataNode"

        else:
            msg = "Insufficient args for DataNode construction."
            raise Exception(msg)

        super().__init__()

    @property
    def label(self):
        return self._label

    @property
    def full_label(self):
        return self.class_node.label + '.' + self._label

    def __repr__(self):
        return "DataNode({}, {})".format(self.class_node, self.label)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.label == other.label) \
               and (self.prefix == other.prefix) \
               and (self.class_node.label == other.class_node.label)

    def __hash__(self):
        return hash((self.label, self.prefix, self.class_node))


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
               and (self.prefix == other.prefix)

    def __hash__(self):
        return hash((self.label, self.prefix))


class DataLink(SSDLink):
    """
    Link between ClassNode -> DataNode
    """
    def __init__(self, label, prefix):
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
    def __init__(self, label, prefix):
        super().__init__(label, prefix)
        self.type = "ColumnLink"

    def __repr__(self):
        return "ColumnLink({})".format(self.label)


class ClassInstanceLink(DataLink):
    """
    Link between ClassNode -> DataNode
    """
    def __init__(self, label, prefix):
        super().__init__(label, prefix)
        self.type = "ClassInstanceLink"

    def __repr__(self):
        return "ClassLink({})".format(self.label)
