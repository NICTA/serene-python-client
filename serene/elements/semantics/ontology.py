"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Defines the Ontology object
"""
import logging
import os.path
import tempfile
import pandas as pd
import itertools as it
import rdflib

from collections import defaultdict
from .base import BaseSemantic
from serene.elements import Class, DataProperty, ObjectProperty
from serene.utils import gen_id

_logger = logging.getLogger()
_logger.setLevel(logging.WARN)


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

        :param file: The name of the .ttl file.
        """
        super().__init__()

        self._name = ""
        self._prefixes = {}
        self.path = None
        self.id = None
        self._uri = "http://www.semanticweb.org/serene/{}".format(gen_id(k=8))
        self._stored = False
        self.description = ""
        self.date_created = None
        self.date_modified = None
        self.source_file = None  # the original file the user starts with (if any)

        # update to a default ID...
        self._update_id(gen_id())

        # if a file is presented, we use this to update
        if file is not None:
            _logger.debug("Importing {} from file.".format(file))

            if os.path.exists(file):

                # attempt to extract an ontology from turtle
                # and load into self...
                try:
                    RDFReader().to_ontology(file, self)
                except Exception:
                    msg = "Failed to read ontology file {}".format(file)
                    raise Exception(msg)

                self.source_file = file

                self.set_filename(os.path.basename(file))
            else:
                msg = "Failed to find file {}".format(file)
                raise FileNotFoundError(msg)
        else:
            # default prefixes...
            self._prefixes = {
                '': '{}#'.format(self._uri + '#'),
                'owl': 'http://www.w3.org/2002/07/owl#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'xsd': 'http://www.w3.org/2001/XMLSchema#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }

            self.set_filename("{}.ttl".format(self.id))

        # default filename
        self._filename = os.path.basename(self.path)

    def _unstore(self):
        """
        Use this once a change is made to the local Ontology that is not
        reflected on the server
        :return: None
        """
        self.source_file = None
        self._stored = False

    def add_class_node(self, node, add_data_nodes=True):
        """Unstore if a class node is added"""
        self._unstore()
        return super().add_class_node(node, add_data_nodes)

    def add_link(self, link):
        """Unstore if a link is added"""
        self._unstore()
        return super().add_link(link)

    def remove_node(self, node):
        """Unstore if a class node is removed"""
        self._unstore()
        return super().add_class_node(node)

    def remove_link(self, link):
        """Unstore if a link is removed"""
        self._unstore()
        return super().add_link(link)

    def _update_id(self, id_var):
        """
        Updates with the latest id string
        :param id_var: The identifier string
        :return:
        """
        self.id = id_var

    def set_filename(self, value):
        """
        Sets the filename for the owl file. This will also update the temporary file cache.
        :param value:
        :return:
        """
        self._name = value
        self.path = os.path.join(
            tempfile.gettempdir(),
            self._name
        )
        return self

    def update(self, json):
        """
        Updates parameters from the server.

        :param json:
        :return:
        """
        self._stored = True
        self.set_filename(json['name'])
        self.description = json['description']
        self.date_created = json['dateCreated']
        self.date_modified = json['dateModified']
        self._update_id(json['id'])
        return self

    def to_turtle(self, filename=None):
        """
        Writes the turtle file to filename, or back to the source
        file if requested...

        :param filename:
        :return:
        """
        fname = filename if filename is not None else self.path
        turtle_string = RDFWriter().to_turtle(self)
        with open(fname, 'w') as f:
            f.write(turtle_string)
            _logger.info("File written to: {}".format(fname))

        return fname

    def prefix(self, prefix, ns):
        """
        Adds a prefixed namespace to the ontology

        :param prefix:
        :param ns: namespace
        :return: The ontology object
        """
        _logger.debug("Adding prefix with prefix={}, uri={}".format(prefix, ns))
        self._prefixes[prefix] = ns
        return self

    def uri(self, uri_string):
        """
        Adds the URI to the ontology
        :param uri_string:
        :return: The ontology object
        """
        _logger.debug("Adding URI string uri={}".format(uri_string))
        self._uri = uri_string
        # update the prefix...
        self._prefixes[''] = '{}#'.format(self._uri)
        return self

    @property
    def namespace(self):
        return self._prefixes['']

    @property
    def prefixes(self):
        return self._prefixes

    @property
    def stored(self):
        """Flag to test whether this ontology is stored on the server."""
        return self._stored

    @property
    def name(self):
        return self._name

    def __repr__(self):
        """
        String output for the ontology...
        :return:
        """
        if not self._stored:
            return "Ontology(local, {})".format(self._name)
        else:
            return "Ontology({}, {})".format(self.id, self._name)


class RDFReader(object):
    """
        Converts RDF objects...
    """
    def __init__(self):
        self.TYPE_MAP = {
            rdflib.RDFS.Literal: str,
            rdflib.XSD.dateTime: pd.datetime,
            rdflib.XSD.unsignedInt: int,
            rdflib.XSD.integer: int,
            rdflib.XSD.string: str,
            rdflib.XSD.boolean: bool,
            rdflib.XSD.float: float,
            rdflib.XSD.double: float,
            rdflib.XSD.date: pd.datetime
        }

    @staticmethod
    def label(node):
        """
        Strips off the prefix in the URI to give the label...

        :param node: The full URI string
        :return: (prefix, label) as (string, string)
        """
        if '#' in node:
            name = node.split("#")[-1]
        else:
            # there must be no # in the prefix e.g. schema.org/
            name = node.split("/")[-1]
        return name

    @staticmethod
    def prefix(node):
        """
        Strips off the name in the URI to give the prefixlabel...

        :param node: The full URI string
        :return: (prefix, label) as (string, string)
        """
        if '#' in node:
            name = node.split("#")[-1]
        else:
            # there must be no # in the prefix e.g. schema.org/
            name = node.split("/")[-1]
        return node[:-len(name)]

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

    def _domains_ranges(self, g):
        """
        Extract all classes which are used either as subject or object in ObjectProperties.
        Add subclass stuff as well.

        :param g: The RDFLib graph object
        :return: List of 3-tuples with class-label-class
        """
        domains_ranges = set()

        object_properties = self.subjects(g, object=rdflib.OWL.ObjectProperty)

        for link in object_properties:

            sources = [s for s in self.objects(g,
                                   subject=link,
                                   predicate=rdflib.RDFS.domain)
                       if issubclass(type(s),rdflib.term.URIRef)]
            destinations = [s for s in self.objects(g,
                                        subject=link,
                                        predicate=rdflib.RDFS.range)
                            if issubclass(type(s),rdflib.term.URIRef)]

            domains_ranges = domains_ranges.union(sources + destinations)

        subclass = [(s, o) for s, p, o in g.triples((None, rdflib.RDFS.subClassOf, None)) if
         (issubclass(type(s), rdflib.term.URIRef) and issubclass(type(o), rdflib.term.URIRef))]

        domains_ranges = domains_ranges.union([s for s,o in subclass])
        domains_ranges = domains_ranges.union([o for s, o in subclass])

        return list(domains_ranges)

    def _extract_links(self, g):
        """
        Extracts the links from the RDFLib ontology graph g. Returns a
        list of 3-tuples with (ClassNode1, label, ClassNode2)

        :param g: The RDFLib graph object
        :return: List of 3-tuples with class-label-class
        """
        all_links = []

        object_properties = self.subjects(g, object=rdflib.OWL.ObjectProperty)

        for link in object_properties:

            sources = self.objects(g,
                                   subject=link,
                                   predicate=rdflib.RDFS.domain)
            destinations = self.objects(g,
                                        subject=link,
                                        predicate=rdflib.RDFS.range)

            links = it.product(sources, destinations)

            links = [(src, dst) for src, dst in links
                     if (issubclass(type(src), rdflib.term.URIRef)
                     and issubclass(type(dst), rdflib.term.URIRef))]

            for x, y in links:
                all_links.append((self.label(x),
                                  self.label(link),
                                  self.label(y),
                                  self.prefix(link)))
        return all_links

    @staticmethod
    def _extract_subclasses(g):
        """

        :param g:
        :return:
        """
        return {s: o for s, p, o in g.triples((None, rdflib.RDFS.subClassOf, None)) if
                (issubclass(type(s), rdflib.term.URIRef) and issubclass(type(o),rdflib.term.URIRef))}

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
        data_properties = self.subjects(g, object=rdflib.OWL.DatatypeProperty)

        for dp in data_properties:

            sources = self.objects(g,
                                   subject=dp,
                                   predicate=rdflib.RDFS.domain)
            destinations = self.objects(g,
                                        subject=dp,
                                        predicate=rdflib.RDFS.range)

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
                        uri,
                        class_nodes,
                        data_node_table,
                        all_links,
                        subclasses,
                        namespaces):
        """
        Builds up the ontology from the node and link elements...

        :param ontology: the initial ontology starting point
        :param uri:
        :param class_nodes: The list of class nodes from the file
        :param data_node_table: The type table for the data_nodes in the file
        :param all_links: The links for the file
        :param subclasses: The links that have a parent class
        :return:
        """
        # first set the uri
        ontology.uri(uri)

        # extract the parents and children, we need to ensure that
        # the parents are created first.
        nodes = self._ordered_classes(class_nodes, subclasses)

        for cls in nodes:
            if cls in subclasses:
                parent = self.label(subclasses[cls])
            else:
                parent = None

            ontology.owl_class(self.label(cls),
                               data_node_table[cls],
                               prefix=self.prefix(cls),
                               is_a=parent)

        # now we add all the links...
        for src, link, dst, prefix in all_links:
            ontology.link(src, link, dst, prefix=prefix)

        # ... and all the prefixes...
        for prefix, ns in namespaces:
            ontology.prefix(prefix, ns)

        return ontology

    def _extract_uri(self, g):
        """Extracts the base URI from the Ontology"""
        candidates = self.subjects(g, object=rdflib.OWL.Ontology)

        if len(candidates) != 1:
            msg = "Failed to read ontology URI from file. {} found".format(candidates)
            _logger.warn("Ontology has no uri!")
            return None
            # raise Exception(msg)

        return str(candidates[0])

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
        # we guess format
        fmt = rdflib.util.guess_format(filename)
        logging.debug("Loading ontology {} with the format {}".format(filename, fmt))
        g.load(filename, format=fmt)

        # get class nodes - we need to leave only those which have URIs
        # blank nodes do not have URIs, we leave them for later
        class_nodes = [s for s in self.subjects(g, object=rdflib.OWL.Class) if issubclass(type(s), rdflib.term.URIRef)]
        domain_ranges = self._domains_ranges(g)

        # when reading ontology from file, we add to class nodes all classes which are used as domains or ranges
        # FIXME: this is not completely correct. instead we need an ontology manager which should consider
        # all ontologies as one big
        class_nodes = list(set(class_nodes).union(set(domain_ranges)))

        # build the ontology object...
        ontology = self._build_ontology(ontology=ontology,
                                        uri=self._extract_uri(g),
                                        class_nodes=class_nodes,
                                        data_node_table=self._extract_data_nodes(g),
                                        all_links=self._extract_links(g),
                                        subclasses=self._extract_subclasses(g),
                                        namespaces=g.namespaces())

        return ontology


class RDFWriter(object):
    """

    """
    def __init__(self):
        """

        """
        self.TYPE_MAP = {
            str: rdflib.RDFS.Literal,
            int: rdflib.XSD.integer,
            bool: rdflib.XSD.boolean,
            float: rdflib.XSD.float,
            pd.datetime: rdflib.XSD.dateTime
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
                rdf(cls.label),
                rdflib.RDF.type,
                rdflib.OWL.Class
            ))

            if cls.parent is not None:
                g.add((
                    rdf(cls.label),
                    rdflib.RDFS.subClassOf,
                    rdf(cls.parent.label)
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
                rdf(link.label),
                rdflib.RDF.type,
                rdflib.OWL.ObjectProperty
            ))
            g.add((
                rdf(link.label),
                rdflib.RDFS.domain,
                rdf(link.src.label)
            ))
            g.add((
                rdf(link.label),
                rdflib.RDFS.range,
                rdf(link.dst.label)
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
                rdf(node.label),
                rdflib.RDF.type,
                rdflib.OWL.DatatypeProperty
            ))
            g.add((
                rdf(node.label),
                rdflib.RDFS.domain,
                rdf(node.parent.label)
            ))
            g.add((
                rdf(node.label),
                rdflib.RDFS.range,
                self.TYPE_MAP[node.dtype]
            ))

    def to_turtle(self, ontology):
        """
        Convert the Ontology object to a Turtle RDF string
        :param ontology: The Ontology object
        :return: String of Turtle rdf
        """
        g = rdflib.Graph()

        # add the prefix table
        for prefix, uri in ontology.prefixes.items():
            g.bind(prefix, rdflib.term.URIRef(uri))

        # add the uri...
        g.add((
            rdflib.term.URIRef(ontology.namespace),
            rdflib.RDF.type,
            rdflib.OWL.Ontology))

        # add all the elements...
        self._build_classes(g, ontology)
        self._build_links(g, ontology)
        self._build_data_nodes(g, ontology)

        return g.serialize(format='turtle').decode("utf-8")

