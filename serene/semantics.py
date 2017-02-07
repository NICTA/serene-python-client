"""
 License...
"""
import copy
import itertools as it
import logging
import os.path
import random
import tempfile
from collections import defaultdict

import networkx as nx
import pandas as pd
import rdflib

from .elements import ClassNode, Link, LinkList
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
        if type(nodes) == dict:
            node_dict = nodes
        elif type(nodes) == list:
            node_dict = {k: str for k in nodes}

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
            link = Link("property", node, dataNode)
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
                      .format(source, link, dest))

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
        locate = [x for x in self.links if link == x]
        if len(locate):
            target = locate[0]
            self._graph.remove_edge(target.src, target.dst)
            self._links.remove(link)
        else:
            msg = "Item {} does not exist in the model.".format(link)
            _logger.error(msg)
        return

    def remove_node(self, node):
        """
        Removes a node from the graph...
        :param node: ClassNode to remove
        :return:
        """
        del self._class_table[node.name]
        self._graph.remove_node(node)
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

        id_var = random.randint(1e7, 1e8 - 1)

        self.filename = os.path.join(
            tempfile.gettempdir(),
            "{}.owl".format(id_var)
        )
        self._prefixes = {}
        self._base = "http://www.semanticweb.org/data_integration/{}".format(id_var)
        self.source_file = None

        if file is not None:
            _logger.debug("Importing {} from file.".format(file))

            if os.path.exists(file):

                # attempt to extract an ontology from turtle
                RDFConverter().turtle_to_ontology(file, self)

                self.source_file = file
            else:
                msg = "Failed to find file {}".format(file)
                raise FileNotFoundError(msg)
        else:
            # default prefixes...
            self._prefixes = {
                '': '<{}#>'.format(self._base),
                'owl': '<http://www.w3.org/2002/07/owl#>',
                'rdf': '<http://www.w3.org/1999/02/22-rdf-syntax-ns#>',
                'xsd': '<http://www.w3.org/2001/XMLSchema#>',
                'rdfs': '<http://www.w3.org/2000/01/rdf-schema#>'
            }

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

    def __repr__(self):
        """
        String output for the ontology...
        :return:
        """
        return "Ontology({})".format(self.filename)


class RDFConverter(object):
    """
        Converts RDF objects...
    """
    def __init__(self):
        self.TYPE_LOOKUP = {
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
                property_table[x][self.label(dp)] = self.TYPE_LOOKUP[y]
        return property_table

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
        # the parents are created first...
        children = list(subclasses.keys())
        parents = [c for c in class_nodes if c not in children]

        for cls in parents + children:
            if cls in subclasses:
                parent = self.label(subclasses[cls])
            else:
                parent = None

            ontology.class_node(self.label(cls),
                                data_node_table[self.label(cls)],
                                prefix='',
                                is_a=parent)

        # now we add all the links...
        for src, link, dest in all_links:
            ontology.link(src, link, dest)

        # and all the prefixes...
        for name, uri in namespaces:
            ontology.prefix(name, uri)

        return ontology

    def turtle_to_ontology(self, filename, ontology=None):
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

        # class nodes
        class_nodes = self.subjects(g, object=rdflib.OWL['Class'])

        # a table of the class node properties (data nodes)
        data_node_table = self._extract_data_nodes(g)

        # links...
        all_links = self._extract_links(g)

        # subclass lookup
        subclasses = self._extract_subclasses(g)

        # build the ontology object...
        ontology = self._build_ontology(ontology,
                                        class_nodes,
                                        data_node_table,
                                        all_links,
                                        subclasses,
                                        g.namespaces())

        return ontology


