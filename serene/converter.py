import itertools as it
from collections import defaultdict

import pandas as pd
import rdflib

from .semantics import Ontology


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
        return node.split("#")[-1]

    @staticmethod
    def subjects(g, subject=None, predicate=None, object=None):
        return [s for s, p, o in g.triples((subject, predicate, object))]

    @staticmethod
    def objects(g, subject=None, predicate=None, object=None):
        return [o for s, p, o in g.triples((subject, predicate, object))]

    def turtle_to_ontology(self, filename='tests/resources/owl/dataintegration_report_ontology.owl'):
        #
        # first load the file
        g = rdflib.Graph()
        g.load(filename, format='n3')

        #print()
        #print("Classes")
        #print("=======")
        # class nodes
        class_nodes = self.subjects(g, object=rdflib.OWL['Class'])

        #for cls in class_nodes:
        #    print(self.label(cls))

        #print()
        #print("Class Properties")
        #print("================")

        # links...
        data_properties = self.subjects(g, object=rdflib.OWL['DatatypeProperty'])

        property_table = defaultdict(dict)

        for dp in data_properties:

            sources = self.objects(g, subject=dp, predicate=rdflib.RDFS['domain'])
            destinations = self.objects(g, subject=dp, predicate=rdflib.RDFS['range'])

            links = it.product(sources, destinations)

            for x, y in links:
                property_table[x][self.label(dp)] = self.TYPE_LOOKUP[y]

        #for k, v in property_table.items():
        #    print(self.label(k), "->", v)

        #print()
        #print("Links")
        #print("=====")

        # links...
        object_properties = self.subjects(g, object=rdflib.OWL['ObjectProperty'])
        all_links = []

        for link in object_properties:

            sources = self.objects(g, subject=link, predicate=rdflib.RDFS['domain'])
            destinations = self.objects(g, subject=link, predicate=rdflib.RDFS['range'])

            links = it.product(sources, destinations)

            for x, y in links:
                #print(self.label(x), "-{}->".format(self.label(link)), self.label(y))
                all_links.append((self.label(x), self.label(link), self.label(y)))

        ontology = Ontology()
        for cls in class_nodes:
            ontology = ontology.class_node(self.label(cls), property_table[self.label(cls)])

        for op in all_links:
            ontology = ontology.link(*op)

        return ontology
