"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the SSD dataset description object
"""
import json
import logging
from collections import OrderedDict
from collections import defaultdict

from .base import BaseSemantic
from ..elements import DataNode, Link, TransformList, IdentTransform
from ..elements import Transform, Mapping, Column, ClassNode
from ..dataset import DataSet
from ..semantics.ontology import Ontology
from serene.utils import gen_id, convert_datetime
from serene.visualizers import SSDVisualizer

_logger = logging.getLogger()
_logger.setLevel(logging.WARN)


class SSD(object):
    """
        Semantic source description is the translator between a DataSet
        and a set of Ontologies

        The SemanticSourceDesc contains the data columns, a transform
        layer, a mapping between each column to a DataNode, and a
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
        self._transforms = TransformList()  # to be removed
        self._stored = False  # is stored on the server?
        self._date_created = None
        self._date_modified = None

        # semantic model
        self._semantic_model = BaseSemantic()

        # the mapping object from Column -> Mapping
        # this should always contain the complete Column set,
        # so that the user can easily see what is and isn't
        # yet mapped.
        self._mapping = {}
        if self._dataset is not None:
            for i, column in enumerate(self._dataset.columns):
                self._mapping[column] = Mapping(
                    column,
                    node=None,
                    transform=IdentTransform())

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
        self._semantic_model = reader.semantic_model

        # build up the mapping through the interface...
        for col, ds in reader.mapping:
            self.map(col, ds, _init=False)

        self._semantic_model.summary()

        return self

    def _find_column(self, column):
        """
        Searches for a Column in the system given a abbreviated
        Column type. e.g. Column("name")

        would return the full Column object (if it exists in the system):

        Column("name", index=1, file="something.csv")

        :param column: An abbreviated Column type
        :return: The actual Column in the system, or an error if not found or ambiguous
        """
        col = Column.search(self._dataset.columns, column)
        if col is None:
            msg = "Failed to find column: {}".format(column)
            _logger.error(msg)
            raise Exception(msg)
        return col

    def _find_data_node(self, data_node):
        """
        Searches for a Datanode in the system given a abbreviated
        DataNode type. e.g. DataNode("name")

        would return the full DataNode object (if it exists in the system):

        DataNode("name", parent=ClassNode("Person"))

        :param data_node: An abbreviated DataNode type
        :return: The actual DataNode in the system, or an error if not found or ambiguous
        """
        data_nodes = self._ontology.idata_nodes
        dn = DataNode.search(data_nodes, data_node)
        if dn is None:
            msg = "Failed to find DataNode: {}".format(data_node)
            _logger.error(msg)
            raise Exception(msg)
        return dn

    def _find_transform(self, transform):
        """
        Searches for a Transform in the system given a abbreviated
        Transform type. e.g. Transform(1)

        would return the full Transform object (if it exists in the system):

        Transform(sql="select * from table")

        :param transform: An abbreviated Transform type
        :return: The actual Transform in the system, or an error if not found or ambiguous
        """
        ts = self._transforms
        t = Transform.search(ts, transform)
        if t is None:
            msg = "Failed to find Transform: {}".format(t)
            _logger.error(msg)
            raise Exception(msg)
        return t

    def _find_class(self, cls):
        """
        Searches for a ClassNode in the system given a abbreviated
        ClassNode type. e.g. ClassNode('Person')

        would return the full Link object (if it exists in the system):

        ClassNode("Person", ["name", "dob", "phone-number"])

        :param cls: An abbreviated ClassNode type
        :return: The actual ClassNode in the system, or an error if not found or ambiguous
        """
        classes = set(self._ontology.iclass_nodes)

        c = ClassNode.search(classes, cls)
        if c is None:
            msg = "Failed to find Class: {}".format(c)
            _logger.error(msg)
            raise Exception(msg)
        return c

    def _find_link(self, link):
        """
        Searches for a Link in the system given a abbreviated
        Link type. e.g. Link('owns')

        would return the full Link object (if it exists in the system):

        Link(src=ClassNode("Person"), dst=ClassNode("Car"), relationship='owns')

        :param link: An abbreviated Link type
        :return: The actual Link in the system, or an error if not found or ambiguous
        """
        link_ = Link.search(self.links, link)
        if link_ is None:
            msg = "Failed to find Link: {}".format(link)
            _logger.error(msg)
            raise Exception(msg)
        return link_

    def find(self, item):
        """
        Helper function to locate objects using shorthands e.g.

        ssd.find(Transform(1))

        :param item: Transform, Column, ClassNode, Link or DataNode object
        :return:
        """
        if issubclass(type(item), Transform):
            return self._find_transform(item)
        elif issubclass(type(item), DataNode):
            return self._find_data_node(item)
        elif issubclass(type(item), Column):
            return self._find_column(item)
        elif issubclass(type(item), ClassNode):
            return self._find_class(item)
        elif issubclass(type(item), Link):
            return self._find_link(item)
        else:
            raise TypeError("This type is not supported in find().")

    def map(self, column, node, _init=True):
        """
        Adds a link between the column and data_node for the
        mapping. A transform can also be applied to change
        the column.

        :param column: The source Column
        :param data_node: The destination DataNode
        :param _init: if the mapping comes from the SSDreader, then we should not change nodes
        :return:
        """
        if issubclass(type(column), str):
            column = Column(column)

        if issubclass(type(node), str):
            if '.' in node:
                node = DataNode(*node.split('.'))
            else:
                # TODO: ClassNode should be supported for foreign keys!
                node = ClassNode(node)

        return self._map(column, node, predicted=False, _init=_init)

    def _map(self, column, node, predicted=False, _init=True):
        """
        [Internal] Adds a link between the column and data_node for the
        mapping. A transform can also be applied to change
        the column.

        :param column: The source Column
        :param node: The destination DataNode
        :param predicted: Is the mapping predicted or not (internal)
        :param _init: if the mapping comes from the SSDreader, then we should not change nodes or semantic model
        :return:
        """
        assert type(column) == Column
        assert type(node) == DataNode

        col = self._find_column(column)
        if _init:
            dn = self._find_data_node(node)
        else:
            dn = node

        # add the mapping element to the table...
        self._mapping[col] = Mapping(col, dn, None, predicted)

        # grab the parent from the ontology
        if _init:
            parent = ClassNode.search(self._ontology.iclass_nodes, dn.parent)
        else:
            parent = dn.parent
        if parent is None:
            msg = "{} does not exist in the ontology.".format(parent)
            raise Exception(msg)

        # by making this mapping, the class node is now in the SemanticModel...
        if _init:
            self._semantic_model.add_class_node(parent, add_data_nodes=False)

        target = Link(dn.name, parent, dn)
        link = Link.search(self._ontology.ilinks, target)
        if link is None:
            msg = "{} does not exist in the ontology.".format(target)
            raise Exception(msg)

        self._semantic_model.add_link(target)

        return self

    def _add_transform(self, transform):
        """
        Initialize the transform (if available)
        otherwise we use the identity transform...

        :param transform:
        :return:
        """
        if transform is not None:
            if callable(transform):
                t = Transform(id=None, func=transform)
            else:
                t = self._find_transform(transform)
        else:
            t = IdentTransform()

        self._transforms.append(t)

        return t

    def link(self, src, dst, relationship):
        """
        Adds a link between ClassNodes. The relationship must
        exist in the ontology

        :param src: Source ClassNode
        :param dst: Destination ClassNode
        :param relationship: The label for the link between the src and dst nodes
        :return: Updated SSD
        """
        if issubclass(type(src), str):
            src = ClassNode(src)

        if issubclass(type(dst), str):
            dst = ClassNode(dst)

        s_class = self._find_class(src)
        d_class = self._find_class(dst)

        # now check that the link is in the ontology...
        parent_links = self._ontology.ilinks
        target_link = Link(relationship, s_class, d_class)
        link = Link.search(parent_links, target_link)

        if link is None:
            msg = "Link {} does not exist in the ontology.".format(relationship)
            _logger.error(msg)
            raise Exception(msg)
        elif link in self.links:
            msg = "Link {} is already in the links".format(link)
            _logger.info(msg)
        else:
            target_link.prefix = link.prefix

            # if it is ok, then add the new link to the SemanticModel...
            self._semantic_model.add_link(target_link)

        return self

    def sample(self, column, transform=None, n=10):
        """
        Samples the column

        ssd.sample(Column('name'))

        optionally you can apply the transform to the column...

        ssd.sample(Column('name'), Transform(1))

        :param column: The column object to sample
        :param transform: An optional transform to apply
        :param n: The number of samples to take
        :return: List of samples from the column
        """
        # first we check that this request is valid...
        col = self._find_column(column)

        # next we pull a sample from the Column dataframe...
        samples = col.df[col.name].sample(n).values

        if transform is None:
            return list(samples)
        else:
            # if we have a valid transform, then we can apply it...
            t = self._find_transform(transform)
            return list(map(t.func, samples))

    def remove(self, item):
        """
        Remove the item. The item can be a transform or
        a data node, column or link.

        :return:
        """
        if type(item) == Transform:
            elem = self._find_transform(item)
            self._transforms.remove(elem)

        elif type(item) == DataNode:
            elem = self._find_data_node(item)
            key = None
            value = None
            for k, v in self._mapping.items():
                if v.node == elem:
                    key = k
                    value = v
                    break

            # we put the default mapping back in...
            self._mapping[key] = Mapping(
                value.column,
                node=None,
                transform=IdentTransform()
            )
            # we also need to remove the old transform
            self._transforms.remove(value.transform)

            # note that we now need to check whether
            # we should remove the classNode from the
            # SemanticModel
            old = set(self._semantic_model.class_nodes)
            new = set(self.class_nodes)
            targets = new - old
            for t in targets:
                self._semantic_model.remove_node(t)

        elif type(item) == Column:
            elem = self._find_column(item)
            key = None
            for k, v in self._mapping.items():
                if v.col == elem:
                    key = k
                    break
            del self._mapping[key]

        elif type(item) == Link:
            elem = self._find_link(item)

            self._semantic_model.remove_link(elem)
        else:
            raise TypeError("This type is not supported in remove().")

        return self

    def show(self):
        """
        Shows the Semantic Modeller in a png. This uses the SSDVisualizer helper
        to launch the graphviz image.

        :return: None
        """
        self.summary()
        SSDVisualizer(self).show()
        return

    def summary(self):
        """Prints out the summary string..."""
        print()
        print("{}:".format(self._name))
        print(self._full_str())

    @property
    def name(self):
        """Returns the current name of the SSD"""
        return self._name

    @property
    def id(self):
        """Returns the current ID of the SSD"""
        return self._id

    @property
    def stored(self):
        """The stored flag if the ssd is on the server"""
        return self._stored

    @property
    def version(self):
        """The SSD version"""
        return self._VERSION

    @property
    def ontology(self):
        """The read-only list of ontology objects"""
        return self._ontology

    @property
    def ssd(self):
        """The SSD as a Python dictionary"""
        builder = SSDJsonWriter(self)
        return builder.to_dict()

    @property
    def json(self):
        """The SSD as a JSON string"""
        builder = SSDJsonWriter(self)
        return builder.to_json()

    @property
    def dataset(self):
        """returns the dataset provided"""
        return self._dataset

    @property
    def model(self):
        """The read-only semantic model"""
        return self._semantic_model

    @property
    def predictions(self):
        """The list of all predicted mappings"""
        return list(m for m in self.mappings if m.predicted)

    @property
    def mappings(self):
        """The list of all current mappings"""
        return list(self._mapping.values())

    @property
    def columns(self):
        """The list of columns currently used"""
        return list(self._mapping.keys())

    @property
    def data_nodes(self):
        """All available data nodes"""
        return list(m.node for m in self.mappings)

    @property
    def class_nodes(self):
        """All available class nodes"""
        #return list(set(n.parent for n in self.data_nodes if n is not None))
        return self._semantic_model.class_nodes

    @property
    def data_links(self):
        """ClassNode - DataNode links in the semantic model"""
        # # grab the object links from the model...
        return self._semantic_model.data_links

    @property
    def class_links(self):
        """ClassNode - ClassNode links in the semantic model"""
        # # grab the object links from the model...
        return [link for link in self._semantic_model.links
                if link.link_type == Link.OBJECT_LINK]

    @property
    def links(self):
        """
            The links in the SSD. The links come from 2 sources. The object links come from
            what's defined in the semantic model. The data links come
            from the relevant mappings to the columns. The other links
            are not necessary.
        """
        return self.data_links + self.class_links

    def _full_str(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        map_str = [str(m) for m in self.mappings]
        link_str = [str(link) for link in self.links if link.link_type == Link.OBJECT_LINK]

        items = map_str + link_str
        full_str = '\n\t'.join(items)

        return "[\n\t{}\n]".format(full_str)

    def __repr__(self):
        return "SSD({}, {})".format(self._name, self._dataset.filename)


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
        self._ds_endpoint = dataset_endpoint
        self._on_endpoint = ontology_endpoint
        self._ontology = self._find_ontology(blob)
        self._dataset = self._find_dataset(blob)
        self._semantic_model = self._build_semantic(blob, self._ontology)
        self._mapping = self._build_mapping(blob)

    @property
    def ontology(self):
        return self._ontology

    @property
    def dataset(self):
        return self._dataset

    @property
    def semantic_model(self):
        return self._semantic_model

    @property
    def mapping(self):
        return self._mapping

    def _find_ontology(self, json):
        """Pulls the ontology reference from the SSD and queries the server"""
        ontologies = json['ontology']
        # TODO: Only one ontology used!! Should be a sequence!
        return self._on_endpoint.get(ontologies[0])

    def _find_dataset(self, json):
        """Attempts to grab the dataset out from the json string"""
        # fill out the dataset...
        # jsm = json["semanticModel"]

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

    @staticmethod
    def _get_data_nodes(semantic_json):
        """

        :param semantic_json:
        :return:
        """

        class_table = {n["id"]: ClassNode(name=n["label"], prefix=n["prefix"], idx=n["id"])
                       for n in semantic_json["nodes"]
                       if n["type"] == "ClassNode"}

        data_links = [(n["source"], n["target"], n["label"], n["prefix"])
                      for n in semantic_json["links"]
                      if n["type"] == Link.DATA_LINK]

        # fill out lookups for data node ids
        lookup = {}
        for link in data_links:
            src, dst, name, prefix = link
            class_node = class_table[src]
            lookup[dst] = DataNode(class_node, name, prefix=prefix)

        return lookup

    def _build_mapping(self, json):
        """Builds the mapping from the json string"""
        jsm = json["semanticModel"]
        raw_map = json["mappings"]

        raw_data_nodes = self._get_data_nodes(jsm)

        data_nodes = {dn_id: DataNode.search(self._semantic_model.data_nodes, dn)
                      for (dn_id, dn) in raw_data_nodes.items()}

        column_map = {col.id: col for col in self._dataset.columns}

        values = []
        for link in raw_map:
            cid = column_map[link["attribute"]]
            node = data_nodes[link["node"]]
            values.append((cid, node))

        return values

    @staticmethod
    def _build_semantic(blob, ontology):
        """Builds the semantic model from a json string"""
        sm = BaseSemantic()
        jsm = blob["semanticModel"]
        raw_nodes = jsm["nodes"]
        links = jsm["links"]

        class_table = {n["id"]: ClassNode(n["label"], prefix=n["prefix"], idx=n["id"]) for n in raw_nodes
                       if n["type"] == "ClassNode"}

        # print(">>>> class_table >>>>>", class_table)

        class_links = [(n["source"], n["target"], n["label"], n["prefix"])
                       for n in links
                       if n["type"] == Link.OBJECT_LINK]

        # print("===== class_links ====")
        # for c in class_links:
        #     print(c)

        data_links = [(n["source"], n["target"], n["label"], n["prefix"])
                      for n in links
                      if n["type"] == Link.DATA_LINK]

        # print("+++++ data_links +++++", data_links)

        # fill out lookups for data properties
        lookup = defaultdict(list)
        for link in data_links:
            src, dst, name, prefix = link
            lookup[src].append((dst, name, prefix))

        # print("lookup: ", lookup)

        # Note: we can have class nodes which do not have any data nodes
        class_node_lookup = {}
        for cls in class_table:
            class_node = class_table[cls]
            cn = ClassNode.search(ontology.iclass_nodes, class_node)
            # now we change the data nodes on this class node to the ones which exist within the semantic model
            data_nodes = lookup[cls]
            cn = ClassNode(cn.name, nodes=None, prefix=cn.prefix)
            if len(data_nodes):
                for (dst, name, prefix) in data_nodes:
                    dns = DataNode.search(ontology.idata_nodes, DataNode(cn, name, prefix=prefix))
                    cn.nodes.append(dns)
            # now add this class node together with data nodes
            # data property links will be automatically added
            sm.add_class_node(cn)
            class_node_lookup[cls] = cn  # update class table with the correct class node

        # finally we add the class links
        for link in class_links:
            src, dst, name, prefix = link
            # print("Processing LINK {}, {}, {}, {}".format( src, name, dst, prefix))
            # source class node...
            scn = class_node_lookup[src]
            # dst class node...
            dcn = class_node_lookup[dst]
            # print("ADDING LINK {}, {}, {}, {}".format(scn, name, dcn, prefix))
            sm.link(scn.name, name, dcn.name, prefix=prefix)

        # print("~~~class nodes: ", sm.class_nodes)

        return sm


class SSDJsonWriter(object):
    """
    Helper class to build up the json output for the SSD file.
    """
    def __init__(self, ssd):
        """

        :param ssd:
        """
        self._ssd = ssd

        self._all_nodes = [n for n in
                           self._ssd.class_nodes + self._ssd.data_nodes
                           if n is not None]

        self._node_map = {m: i for i, m in enumerate(self._all_nodes)}
        self._attr_map = {m: i for i, m in enumerate(self._ssd.mappings)}

    def to_dict(self):
        """Builds the dictionary representation of the SSD"""
        d = OrderedDict()
        d["name"] = self._ssd.name
        d["ontologies"] = [self._ssd.ontology.id]
        d["semanticModel"] = self.semantic_model
        d["mappings"] = self.mappings
        return d

    def to_json(self):
        """Builds the complete json object string"""
        return json.dumps(self.to_dict())

    @property
    def semantic_model(self):
        """Builds out the .ssd semantic model section..."""

        return {
            "nodes": [node.ssd_output(index)
                      for index, node in enumerate(self._all_nodes)],
            "links": [link.ssd_output(index, self._node_map)
                      for index, link in enumerate(self._ssd.links)]
        }

    @property
    def mappings(self):
        """BUilds out the .ssd mapping section.."""
        return [
            {
                "attribute": m.column.id,
                "node": self._node_map[m.node]
            } for m in self._ssd.mappings if m.node is not None]