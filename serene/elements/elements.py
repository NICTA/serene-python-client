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
        return id(self)


class Mapping(object):
    """
        A Mapping object that describes the link between a column and
        a data node. An optional transform can be included.
    """
    def __init__(self, column, node=None, transform=None, predicted=False):
        if type(column) != Column:
            raise TypeError("Column type required for 'column' in Mapping object")

        if (node is not None) and (type(node) != DataNode):
            raise TypeError("DataNode type required for 'node' in Mapping object")

        if (transform is not None) and not issubclass(type(transform), Transform):
            raise TypeError("Transform type required for 'transform' in Mapping object: {}".format(transform))

        self.column = column
        self.node = node
        self.transform = transform if transform is not None else IdentTransform()
        self.predicted = predicted

    def __repr__(self):
        if self.transform is not None:
            if type(self.transform) == IdentTransform:
                s = "{} -> {}".format(self.column, self.node)
            else:
                s = "{} -> Transform({}) -> {}".format(self.column, self.transform.id, self.node)
        else:
            s = "{} -> {}".format(self.column, self.node)

        # we add an asterix to denote predicted values...
        if self.predicted:
            s = '{} [*]'.format(s)
        return s


class Transform(Searchable):
    """
        The transform object that describes the transform between
        the raw column and the datanode type.
    """
    _id = 0  # Transform ID value...
    getters = [
        lambda t: t.id,
        lambda t: t.name,
        lambda t: t.sql
    ]

    def __init__(self, id=None, func=None, name='', sql=None):
        if id is None:
            self.id = Transform._id
            Transform._id += 1
        else:
            self.id = id
        self.name = name
        self.func = func
        self.sql = sql if sql is not None else 'SELECT * from {}'.format(name)
        super().__init__()

    @staticmethod
    def apply(col):
        return 'SELECT {} from {}'.format(col.name, col.set_filename)

    def __repr__(self):
        return "{{\n" \
               "    id: {}\n" \
               "    name: {}\n" \
               "    func: {}\n" \
               "    sql: {}\n" \
               "}}".format(self.id, self.name, self.func, self.sql)

    def __hash__(self):
        return id(self)


class IdentTransform(Transform):
    """
    Transform that is just the identity...
    """
    def __init__(self, id=None, func=None, name='', sql=None):
        super().__init__(id, func, name, sql)
        self.name = "ident"
        self.func = lambda x: x

    @staticmethod
    def apply(col):
        return 'SELECT {} from {}'.format(col.name, col.filename)


class TransformList(collections.MutableSequence):
    """
    Container type for Transform objects in the Semantic Source Description

    """
    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), Transform):
            raise TypeError("Only Transform types permitted: {}".format(v))

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
        transforms = []
        for v in self.list:
            s = "Transform({}): {}".format(v.id, v)
            transforms.append(s)
        return '\n'.join(transforms)


class ClassNode(Searchable):
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
        lambda node: node.nodes if len(node.nodes) else None,
        lambda node: node.parent if node.parent else None
    ]

    def __init__(self, name, nodes=None, prefix=None, parent=None):
        """
        A ClassNode is initialized with a name, a list of string nodes
        and optional prefix and/or a parent ClassNode

        :param name: The string name of the ClassNode
        :param nodes: A list of strings to initialze the DataNode objects
        :param prefix: The URI prefix
        :param parent: The parent object if applicable
        """
        self.name = name
        self.prefix = prefix
        self.parent = parent

        if nodes is None:
            self.nodes = []
        elif issubclass(type(nodes), list) or issubclass(type(nodes), types.GeneratorType):
            self.nodes = [DataNode(self, n, dtype=str) for n in nodes]
        else:
            self.nodes = [DataNode(self, n, dtype=dtype) for n, dtype in nodes.items()]

    def ssd_output(self, ident):
        """
        The output function has not
        :return:
        """
        return {
            "id": ident,
            "label": self.name,
            "prefix": self.prefix if self.prefix is not None else "",
            "type": "ClassNode"
        }

    def __repr__(self):
        nodes = [n.name for n in self.nodes]

        if self.parent is None:
            parent = ""
        else:
            parent = ", parent={}".format(self.parent.name)

        return "ClassNode({}, [{}]{})".format(self.name, ", ".join(nodes), parent)

    def __eq__(self, other):
        return (self.name == other.name) \
               and (self.prefix == other.prefix) \
               and (self.nodes == other.nodes)

    def __hash__(self):
        return hash((self.name, self.prefix, frozenset(self.nodes)))


class DataNode(Searchable):
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

    def __init__(self, *names, dtype=str):
        """
        A DataNode is initialized with name and a parent ClassNode object.
        A DataNode can be initialized in the following ways:

        DataNode(ClassNode("Person"), "name")
        DataNode("Person", "name)
        DataNode("name")

        :param names: The name of the parent classnode and the name of the DataNode
        """
        self.dtype = dtype

        if len(names) == 1:
            # initialized with DataNode("name") - for lookups only...
            self.name = names[0]
            self.parent = None

        elif len(names) == 2:
            # here the first element is now the parent...
            parent = names[0]

            if type(parent) == ClassNode:
                # initialized with DataNode(ClassNode("Person"), "name")
                self.parent = parent
            else:
                # initialized with DataNode("Person", "name")
                self.parent = ClassNode(parent)
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
            return "DataNode({}, {})".format(self.parent.name, self.name)
        else:
            return "DataNode({})".format(self.name)

    def __hash__(self):
        if self.parent:
            return hash((self.parent.name, self.name))
        else:
            return hash(self.name)


class Link(Searchable):
    """
        A Link is a relationship between ClassNode->ClassNode or ClassNode->DataNode.
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
        lambda node: Link.node_match(node)
    ]

    # special link names...
    SUBCLASS = "subclass"
    OBJECT_LINK = "ObjectProperty"
    DATA_LINK = "DataProperty"

    def __init__(self, name, src=None, dst=None):
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
        if type(self.dst) == ClassNode:
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
        return {
            "id": index,
            "source": index_map[self.src],
            "target": index_map[self.dst],
            "label": self.name,
            "type": self.link_type
        }

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


class LinkList(collections.MutableSequence):
    """
    Container type for Link objects in the Semantic Source Description
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not isinstance(v, Link):
            raise TypeError("Only Link types permitted")

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
