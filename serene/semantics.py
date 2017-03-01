"""
 License...
"""
import itertools as it
import logging
import os.path
import random
import tempfile
import string
from collections import defaultdict

import networkx as nx
import pandas as pd
import rdflib

from .elements import ClassNode, DataNode, Link, LinkList
from .visualizers import BaseVisualizer

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class BaseSemantic(object):
    """
    Semantic Graph is the base object to construct Semantic objects.
    The object allows Class Nodes, DataNodes and Links to be members.
    """
    def __init__(self):
        """

        :param file:
        """
        self._uri = ""
        self._graph = nx.DiGraph()
        self._class_table = {}
        self._links = LinkList()
        self._LINK = "relationship"
        self._DATA_NODE_LINK = "property"

    def class_node(self, name, nodes=None, prefix=None, is_a=None):
        """
        This function creates a ClassNode on the semantic object.

        :param name:
        :param nodes:
        :param prefix:
        :param is_a: The parent object
        :return: The ontology object
        """
        _logger.debug("Adding class node name={}, "
                      "nodes={}, prefix={}, parent={}"
                      .format(name, nodes, prefix, is_a))

        # if the parent does not exist, refuse to create the class
        parent_class = None
        if is_a is not None:
            if is_a not in self._class_table:
                raise Exception("Failed to create class {}. "
                                "Parent class '{}' does not exist"
                                .format(name, is_a))
            else:
                parent_class = self._class_table[is_a]

        # build up the node list...
        node_dict = {}

        if issubclass(type(nodes), dict):
            node_dict = nodes
        elif issubclass(type(nodes), list):
            assert(len(nodes) == len(set(nodes)))
            node_dict = {k: str for k in nodes}
        else:
            msg = "Unknown nodes argument: {}".format(nodes)
            raise Exception(msg)

        # now we create a class node object...
        cn = ClassNode(name, node_dict, prefix, parent_class)

        self.add_class_node(cn)

        return self

    def add_class_node(self, node):
        """
        Adds the ClassNode node into the Semantic Model

        :param node:
        :return:
        """
        # if the name is in the class table, then we
        # need to remove it from the graph...
        self._class_table[node.name] = node

        # now add the data property links
        for dataNode in node.nodes:
            link = Link(self._DATA_NODE_LINK, node, dataNode)
            self.add_link(link)

        self._graph.add_node(node)

    def link(self, source, link, dest):
        """
        This function adds a Link relationship between two class nodes
        in the SemanticBase object. The ClassNode source and dest must
        already exist in the SemanticBase graph. This method will create
        the Link object.

        :param source: The source ClassNode
        :param link: The name of the Link relationship
        :param dest: The destination ClassNode object
        :return: The ontology object
        """
        _logger.debug("Adding relationship between "
                      "classes {} and {} as '{}'"
                      .format(source, dest, link))

        if source not in self._class_table:
            msg = "Item {} is not in the class nodes".format(source)
            raise Exception(msg)
        elif dest not in self._class_table:
            msg = "Item {} is not in the class nodes".format(dest)
            raise Exception(msg)

        src = self._class_table[source]
        dst = self._class_table[dest]
        link = Link(link, src, dst)

        self.add_link(link)

        return self

    def add_link(self, link):
        """

        :return:
        """
        self._links.append(link)

        # we also add the link to the graph...
        self._graph.add_edge(
            link.src,
            link.dst,
            {self._LINK: link})
        return self

    def find_class_node(self, name):
        """

        :param name:
        :return:
        """
        ct = self._class_table
        if name in ct:
            return ct[name]
        else:
            return None

    def remove_link(self, link):
        """
        Removes a link from the graph...
        :param link:
        :return:
        """
        _logger.debug("Attempting to remove link: {}".format(link))
        target = Link.search(self.links, link)
        if target:
            _logger.debug("Found true link: {}".format(target))
            try:
                # we need to be careful, because if the nodes
                # are destroyed, then the edge will not exist!
                self._graph.remove_edge(target.src, target.dst)
                _logger.debug("Edge removed successfully {}-{}".format(target.src, target.dst))
            except Exception:
                _logger.debug("Missing edge, nodes already removed. {}".format(link))
                pass

            self._links.remove(target)
            _logger.debug("Removed link: {}".format(target))
        else:
            msg = "Item {} does not exist in the model.".format(link)
            _logger.error(msg)
        return

    @staticmethod
    def _is_node(value):
        """
        Determines whether the value is a ClassNode or a DataNode

        :param value:
        :return:
        """
        return issubclass(type(value), ClassNode) or issubclass(type(value), DataNode)

    @staticmethod
    def _is_link(value):
        """
        Determines whether the value is a Link

        :param value:
        :return:
        """
        return issubclass(type(value), Link)

    def remove_node(self, node):
        """
        Removes a node from the graph...
        :param node: ClassNode to remove
        :return:
        """
        _logger.debug("remove_node called with {}".format(node))
        if not self._is_node(node):
            msg = "{} is not a ClassNode or DataNode".format(node)
            raise Exception(msg)

        elif issubclass(type(node), ClassNode):
            _logger.debug("Removing ClassNode: {}".format(node))
            # first we find the actual reference...
            true_node = self._class_table[node.name]

            _logger.debug("Actual node found: {}".format(true_node))

            # recursively remove the DataNodes...
            for n in true_node.nodes:
                self.remove_node(n)

            # next we remove the links pointing to that node.
            # we need to be careful when iterating as we are deleting
            # the list as we go!!
            for link in self.links[::-1]:
                # print()
                # print(">>>>>>>>>> {} <<<<<<<<<<<".format(link.name))
                # print()
                # print(">>> Searching link: {}".format(link))
                # print(">>>   which has {}".format(link.dst))
                # print(">>>   which has {}".format(link.src))
                # print(">>>   comparing {}".format(true_node))
                # print(">>>        with {}".format(link.dst))
                # print(">>>         and {}".format(link.src))
                if link.dst == true_node or link.src == true_node:
                    _logger.debug("Link found, removing {}".format(link))
                    self.remove_link(link)

            # remove the ClassNode from the class table
            del self._class_table[true_node.name]
        else:
            _logger.debug("Removing DataNode: {}".format(node))

            # first we find the actual reference...
            true_node = DataNode.search(self.data_nodes, node)

            _logger.debug("Actual node found: {}".format(true_node))

        self._graph.remove_node(true_node)

        del true_node

        return

    @staticmethod
    def flatten(xs):
        return [x for y in xs for x in y]

    @property
    def class_nodes(self):
        """The class node objects in the graph"""
        return list(self._class_table.values())

    @property
    def data_nodes(self):
        """All of the data node objects in the graph"""
        # we grab the data nodes from the class nodes...
        # make sure we don't return any duplicates...
        data_nodes = set(self.flatten(cn.nodes for cn in self.class_nodes))

        return list(data_nodes)

    @property
    def links(self):
        """Returns all the links in the graph"""
        # links = nx.get_edge_attributes(self._graph, self._LINK)
        # return list(links.values())
        return self._links

    @property
    def class_links(self):
        """Returns all the links in the graph"""
        # links = nx.get_edge_attributes(self._graph, self._LINK)
        # return list(links.values())
        return [link for link in self._links if link.name != self._DATA_NODE_LINK]

    def summary(self):
        """Prints a summary of the ontology"""
        print("Class Nodes:")
        for cls in self.class_nodes:
            print("\t", cls)
        print()

        print("Data Nodes:")
        for dn in self.data_nodes:
            print("\t", dn)
        print()

        print("Links:")
        for link in self.links:
            print("\t", link)
        print()

    def show(self):
        BaseVisualizer(self).show()
        return


class Ontology(BaseSemantic):
    """
        The Ontology object holds an ontolgy. This can be specified from
        an OWL file, or built by hand using the SemanticBase construction
        methods. Note that this is a subset of the OWL specification, and
        only ClassNodes, DataNodes, relationships and subclass properties
        are used. All other features are ignored.

        This class extends the SemanticBase object to include Ontology
        specific elements such as loading from a file, prefixes and a
        URI value.

    """
    def __init__(self, file=None):
        """
        The constructor can take an additional file object, which
        will be read to build links and class nodes. Note that not
        all attributes from an OWL file will be imported.

        :param file: The name of the .owl file.
        """
        super().__init__()

        id_var = self._rand_id()

        self.filename = os.path.join(
            tempfile.gettempdir(),
            "{}.owl".format(id_var)
        )
        self._prefixes = {}
        self._base = "http://www.semanticweb.org/serene/{}".format(id_var)
        self.source_file = None
        self.id = id_var
        self._stored = False

        if file is not None:
            _logger.debug("Importing {} from file.".format(file))

            if os.path.exists(file):

                # attempt to extract an ontology from turtle
                # and load into self...
                RDFReader().to_ontology(file, self)

                self.source_file = file
            else:
                msg = "Failed to find file {}".format(file)
                raise FileNotFoundError(msg)
        else:
            # default prefixes...
            self._prefixes = {
                '': '{}#'.format(self._base),
                'owl': 'http://www.w3.org/2002/07/owl#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'xsd': 'http://www.w3.org/2001/XMLSchema#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }

    @staticmethod
    def _rand_id():
        """
        Generates a random temporary id. Note that this is alpha-numeric.
        Alpha character ids are used to indicate that this ontology is not
        stored.
        Integer numbers indicate that this ontology object is stored on the
        server.

        :return:
        """
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(16))

    def set_stored_flag(self, stored):
        self._stored = stored

    def to_turtle(self, filename=None):
        """
        Writes the turtle file to filename, or back to the source
        file if requested...

        :param filename:
        :return:
        """
        fname = filename if filename is not None else self.source_file
        turtle_string = RDFWriter().to_turtle(self)
        with open(fname) as f:
            f.write(turtle_string)
            _logger.info("File written to: {}".format(fname))

        return fname

    def prefix(self, prefix, uri):
        """
        Adds a URI prefix to the ontology

        :param prefix:
        :param uri:
        :return: The ontology object
        """
        _logger.debug("Adding prefix with prefix={}, uri={}".format(prefix, uri))
        self._prefixes[prefix] = uri
        return self

    def uri(self, uri_string):
        """
        Adds the URI to the ontology
        :param uri_string:
        :return: The ontology object
        """
        _logger.debug("Adding URI string uri={}".format(uri_string))
        self._uri = uri_string
        return self

    @property
    def prefixes(self):
        return self._prefixes

    def __repr__(self):
        """
        String output for the ontology...
        :return:
        """
        if self.id is not None:
            return "Ontology({}, {})".format(self.id, self.filename)
        else:
            return "Ontology({})".format(self.filename)


class RDFReader(object):
    """
        Converts RDF objects...
    """
    def __init__(self):
        self.TYPE_MAP = {
            rdflib.RDFS['Literal']: str,
            rdflib.XSD['dateTime']: pd.datetime,
            rdflib.XSD['unsignedInt']: int,
            rdflib.XSD['integer']: int,
            rdflib.XSD['string']: str,
            rdflib.XSD['boolean']: bool,
            rdflib.XSD['float']: float,
            rdflib.XSD['double']: float,
            rdflib.XSD['date']: pd.datetime
        }

    @staticmethod
    def label(node):
        """
        Strips off the prefix in the URI to give the label...

        :param node: The full URI string
        :return: string
        """
        if '#' in node:
            return node.split("#")[-1]
        else:
            # there must be no # in the prefix e.g. schema.org/
            return node.split("/")[-1]

    @staticmethod
    def subjects(g, subject=None, predicate=None, object=None):
        """
        Extracts all the 'subject' values from an RDFLib triplet graph g. A filter
        can be made on the subject, predicate or object arguments.

        :param g: The RDFLib graph
        :param subject: Filter for the subjects
        :param predicate: Filter for the predicates
        :param object: Filter for the objects
        :return: List of RDFLib subject values
        """
        return [s for s, p, o in g.triples((subject, predicate, object))]

    @staticmethod
    def objects(g, subject=None, predicate=None, object=None):
        """
        Extracts all the 'object' values from an RDFLib triplet graph g. A filter
        can be made on the subject, predicate or object arguments.

        :param g: The RDFLib graph
        :param subject: Filter for the subjects
        :param predicate: Filter for the predicates
        :param object: Filter for the objects
        :return: List of RDFLib object values
        """
        return [o for s, p, o in g.triples((subject, predicate, object))]

    def _extract_links(self, g):
        """
        Extracts the links from the RDFLib ontology graph g. Returns a
        list of 3-tuples with (ClassNode1, label, ClassNode2)

        :param g: The RDFLib graph object
        :return: List of 3-tuples with class-label-class
        """
        all_links = []

        object_properties = self.subjects(g, object=rdflib.OWL['ObjectProperty'])

        for link in object_properties:

            sources = self.objects(g,
                                   subject=link,
                                   predicate=rdflib.RDFS['domain'])
            destinations = self.objects(g,
                                        subject=link,
                                        predicate=rdflib.RDFS['range'])

            links = it.product(sources, destinations)

            for x, y in links:
                all_links.append((self.label(x),
                                  self.label(link),
                                  self.label(y)))
        return all_links

    @staticmethod
    def _extract_subclasses(g):
        """

        :param g:
        :return:
        """
        return {s: o for s, p, o in g.triples((None, rdflib.RDFS['subClassOf'], None))}

    def _extract_data_nodes(self, g):
        """
        Extracts a dictionary of DataTypeProperties, as in the data nodes
        for a class node. The format is:
            {
                ClassNode1: {
                    DataNode1: type1,
                    DataNode2: type2,
                    ...
                },
                ClassNode2: {
                    DataNode1: type1,
                    DataNode3: type3
                }
                ...
            }

        :param g: The RDFLib graph object
        :return: Nested dictionary of ClassNode -> DataNode -> Python type
        """
        property_table = defaultdict(dict)

        # links...
        data_properties = self.subjects(g, object=rdflib.OWL['DatatypeProperty'])

        for dp in data_properties:

            sources = self.objects(g,
                                   subject=dp,
                                   predicate=rdflib.RDFS['domain'])
            destinations = self.objects(g,
                                        subject=dp,
                                        predicate=rdflib.RDFS['range'])

            links = it.product(sources, destinations)

            for x, y in links:
                property_table[x][self.label(dp)] = self.TYPE_MAP[y]

        return property_table

    def _ordered_classes(self, all_nodes, lookup):
        """
        Returns the nodes in all_nodes in order of increasing depth, given
        the class heirarchy in lookup (a node->parent dictionary)

        :param all_nodes: The list of class nodes
        :param lookup: A class->parent dictionary...
        :return:
        """
        MAX_ITER = 100

        def depth(node, count=0):
            if count > MAX_ITER:
                raise Exception("Cycle found in class node heirarchy")
            elif node not in lookup:
                return count
            else:
                return depth(lookup[node], count + 1)

        depths = [(node, depth(node)) for node in all_nodes]

        s_depths = sorted(depths, key=lambda x: x[1])

        return [x[0] for x in s_depths]

    def _build_ontology(self,
                        ontology,
                        class_nodes,
                        data_node_table,
                        all_links,
                        subclasses,
                        namespaces):
        """
        Builds up the ontology from the node and link elements...

        :param ontology: the initial ontology starting point
        :param class_nodes: The list of class nodes from the file
        :param data_node_table: The type table for the data_nodes in the file
        :param all_links: The links for the file
        :param subclasses: The links that have a parent class
        :return:
        """
        # extract the parents and children, we need to ensure that
        # the parents are created first.
        nodes = self._ordered_classes(class_nodes, subclasses)

        for cls in nodes:
            if cls in subclasses:
                parent = self.label(subclasses[cls])
            else:
                parent = None

            ontology.class_node(self.label(cls),
                                data_node_table[cls],
                                prefix='',
                                is_a=parent)

        # now we add all the links...
        for src, link, dst in all_links:
            ontology.link(src, link, dst)

        # ... and all the prefixes...
        for name, uri in namespaces:
            ontology.prefix(name, uri)

        return ontology

    def to_ontology(self, filename, ontology=None):
        """
        Converts an OWL file from the Turtle format into the Ontology type

        :param filename: OWL or TTL file in Turtle RDF format
        :param ontology: The output ontology object on which to add new classes and nodes
        :return: Ontology object
        """
        # start a new ontology if there is nothing there...
        if ontology is None:
            ontology = Ontology()

        # first load the file
        g = rdflib.Graph()
        g.load(filename, format='n3')

        # build the ontology object...
        ontology = self._build_ontology(ontology,
                                        self.subjects(g, object=rdflib.OWL['Class']),
                                        self._extract_data_nodes(g),
                                        self._extract_links(g),
                                        self._extract_subclasses(g),
                                        g.namespaces())

        return ontology


class RDFWriter(object):
    """

    """
    def __init__(self):
        """

        """
        self.TYPE_MAP = {
            str: rdflib.RDFS['Literal'],
            int: rdflib.XSD['integer'],
            bool: rdflib.XSD['boolean'],
            float: rdflib.XSD['float'],
            pd.datetime: rdflib.XSD['dateTime']
        }

    def _build_classes(self, g, ontology):
        """

        :param g:
        :param ontology:
        :return:
        """

        def rdf(value):
            return self.rdf_node(value, ontology)

        for cls in ontology.class_nodes:
            g.add((
                rdf(cls.name),
                rdflib.RDF.type,
                rdflib.OWL['Class']
            ))

            if cls.parent is not None:
                g.add((
                    rdf(cls.name),
                    rdflib.RDFS.subClassOf,
                    rdf(cls.parent.name)
                ))

    @staticmethod
    def rdf_node(value, ontology, prefix=None):
        """

        :param value:
        :param ontology:
        :param prefix:
        :return:
        """
        if prefix is None:
            default = rdflib.Namespace(ontology.prefixes[''])
        else:
            default = rdflib.Namespace(ontology.prefixes[prefix])

        return rdflib.term.URIRef(default[value])

    def _build_links(self, g, ontology):
        """

        :param g:
        :param ontology:
        :return:
        """
        def rdf(value):
            return self.rdf_node(value, ontology)

        for link in ontology.class_links:
            g.add((
                rdf(link.name),
                rdflib.RDF.type,
                rdflib.OWL['ObjectProperty']
            ))
            g.add((
                rdf(link.name),
                rdflib.RDFS.domain,
                rdf(link.src.name)
            ))
            g.add((
                rdf(link.name),
                rdflib.RDFS.range,
                rdf(link.dst.name)
            ))

    def _build_data_nodes(self, g, ontology):
        """

        :param g:
        :param ontology:
        :return:
        """
        def rdf(value):
            return self.rdf_node(value, ontology)

        for node in ontology.data_nodes:
            g.add((
                rdf(node.name),
                rdflib.RDF.type,
                rdflib.OWL['DatatypeProperty']
            ))
            g.add((
                rdf(node.name),
                rdflib.RDFS.domain,
                rdf(node.parent.name)
            ))
            g.add((
                rdf(node.name),
                rdflib.RDFS.range,
                self.TYPE_MAP[node.dtype]
            ))

    def to_turtle(self, ontology):
        """

        :param ontology:
        :return:
        """
        g = rdflib.Graph()

        for prefix, uri in ontology.prefixes.items():
            g.bind(prefix, rdflib.term.URIRef(uri))

        self._build_classes(g, ontology)
        self._build_links(g, ontology)
        self._build_data_nodes(g, ontology)

        return g.serialize(format='turtle').decode("utf-8")

