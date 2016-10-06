import os
import logging
import pandas as pd
import json
import rdflib
from collections import OrderedDict
from source.semantic_model import SemanticModel
import sys
from source.columns import SourceColumn, ExtendedColumn, SemanticType
from rdflib.namespace import RDF, RDFS, OWL, XSD

# function to convert rdflib graph to networkx
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

########################################################################################
def generate_initial_sd(fname, sourceId, ontology_uri):
    """
    Generate initial source description.
    :param fname: name of the source
    :param sourceId: identifier of the source
    :param ontology_uri: uri of the ontology
    :return: JSON serialized dictionary
    """
    ssd = OrderedDict()
    source = pd.read_csv(fname)

    ssd["version"] = "0.1" # version of the source description
    # identifiers
    ssd["id"] = str(sourceId)
    ssd["name"] = os.path.basename(fname)

    # populate sourceColumns
    ssd["columns"] = [{"id": str(key), "name": col}
                            for key, col in enumerate(source.columns.values.tolist())]

    # populate extendedColumns = transformedColumns = attributes
    ssd["attributes"] = []

    for key, col in enumerate(source.columns.values.tolist()):
        eCol = OrderedDict([("id", str(key)), ("name", col),
                            ("label", "ident"),
                            ("columnIds", [str(key)]),
                            ("sql", "select " + col + " from '" + ssd["name"] + "'")
                            ])
        ssd["attributes"].append(eCol)

    # ontology
    ssd["ontology"] = ontology_uri

    # semantic model
    ssd["semanticModel"] = OrderedDict()
    # should I put additionally uri into node block?
    ssd["semanticModel"]["nodes"] = [
        OrderedDict([("id","0"), ("label",""), ("type","ClassNode")]),
        OrderedDict([("id", "1"), ("label", ""), ("type", "DataNode")])
    ]
    link1 = OrderedDict([("source",""), ("target",""), ("label",""), ("type","ObjectProperty")])
    link2 = OrderedDict([("source", ""), ("target", ""), ("label", ""), ("type", "DataProperty")])
    ssd["semanticModel"]["links"] = [link1,link2]

    # mappings
    ssd["mappings"] = [OrderedDict([("attribute", eCol["id"]),("node","")])
                       for eCol in ssd["attributes"]]



    return json.dumps(ssd, indent=4)


def read_ontology(fname):
    """
    Read ontology serialized in Turtle RDF.
    :param fname: file name where ontology is saved
    :return: RDF graph
    """
    # we can use the returned object to infer paths and other stuff
    g = rdflib.Graph()
    try:
        logging.info("Trying to read ontology in Turtle RDF.")
        g.parse(file=open(fname), format="turtle")
    except Exception as e:
        print(e)
        logging.error("Failed to read ontology in "+ str(fname) + ": " + e.__repr__())
    g.serialize()
    return g

def generate_initial_sds_dir(data_dir, sources_dir, ontol_paths):
    for sId, sPath in enumerate(os.listdir(sources_dir)):
        ssd = generate_initial_sd(os.path.join(sources_dir, sPath), sId, ontol_paths)
        f = os.path.splitext(sPath)[0] + ".ssd"
        f = os.path.join(data_dir, "source_descriptions", f)
        with open(f, "w+") as fwrite:
            fwrite.write(ssd)

def read_sd(fname):
    """
    Read in the source description from the file.
    :param fname: file name where source description is stored
    :return: json-serialized dictionary with source description
    """
    with open(fname) as json_data:
        sd = json.load(json_data)
    # print(type(sd))
    # print(json.dumps(sd,indent=4))
    return sd

########################################################################################

class SourceDescription(object):
    """
    Attributes:

    """
    def __init__(self, ssd):
        """
        Initialize source description.
        :param ssd: dictionary of the source description
        """
        try:
            self.semantic_model = SemanticModel(ssd["semanticModel"])
        except Exception as e:
            logging.error("Failed to initialize the semantic model: "+ str(e))
            sys.exit(1)

        # identifiers
        self.source_id = ssd["id"]
        self.source_name = ssd["name"]

        #sourceColumns
        self.columns = [SourceColumn(col_dict) for col_dict in ssd["columns"]]

        # populate extendedColumns
        self.attributes = [ExtendedColumn(col_dict) for col_dict in ssd["attributes"]]

        # ontology
        self.ontology_uri = ssd["ontology"]
        self.ontologies = [read_ontology(f) for f in ssd["ontology"]]

        # mappings
        self.mappings = ssd["mappings"]

    def convert_graphviz(self, display_id=False):
        """
        Convert the source description into a graphviz Agraph object.
        :param display_id: boolean which indicates whether to include ids into labels
        :return: pygraphviz Agraph object
        """
        # initialize graphviz with semantic model
        g = self.semantic_model.convert_graphviz(display_id)
        g.graph_attr['label'] = self.source_name
        g.graph_attr['fontname'] = 'times:bold'
        # g.graph_attr['fontcolor'] = 'blue'

        # add source columns
        num_nodes = self.semantic_model.graph.number_of_nodes()
        for column in self.columns:
            # ids need to be incremented by num_nodes
            lab = column.name
            if display_id:
                lab = str(column.id) + "\n" + lab
            g.add_node(column.id + num_nodes,
                       label=lab,
                       style="dotted",
                       color='brown',
                       shape='box'
                       )
        g.add_subgraph([col.id + num_nodes for col in self.columns]
                       , rank='same'
                       , name='cluster1'
                       , style='dotted'
                       , color='brown'
                       , fontcolor='brown'
                       , fontname='times'
                       , label='source')

        # add extended columns = attributes
        num_nodes2 = num_nodes + len(self.columns)
        for ecol in self.attributes:
            # ids need to be incremented by num_nodes2
            lab = ecol.name
            if display_id:
                lab = str(ecol.id) + "\n" + lab
            g.add_node(ecol.id + num_nodes2,
                       label=lab,
                       style="rounded",
                       shape='box')
            # add transformation links
            for col_id in ecol.source_columns:
                g.add_edge(ecol.id + num_nodes2
                           , col_id + num_nodes
                           , label=ecol.label
                           , color='brown'
                           , fontname='times-italic'
                           , arrowhead='diamond')

        g.add_subgraph([col.id + num_nodes2 for col in self.attributes]
                       , rank='same'
                       , name='cluster2'
                       , style='dotted'
                       , color='blue'
                       , fontname='times'
                       , fontcolor='blue'
                       , label='transformations'
                       )

        # add mappings
        for map in self.mappings:
            start = int(map["attribute"]) + num_nodes2
            end = int(map["node"])
            g.add_edge(end, start,
                       arrowhead='inv',
                       arrowtail='inv',
                       color='blue')

        return g

    def write_dot(self, path, display_id=False):
        """
        Write the .dot file for the source description using graphviz.
        :param path: string which indicates the location to write the .dot file.
        :param display_id: boolean which indicates whether to include ids into labels
        :return: None
        """
        # get graphviz representation
        g = self.convert_graphviz(display_id)
        # write the .dot file
        g.write(path)
        # g.draw(path+'.png',format='png',prog='dot')

    def write_png(self, path, display_id=False):
        """
        Write the .png file for the source description using graphviz.
        :param path: string which indicates the location to write the .dot file.
        :param display_id: boolean which indicates whether to include ids into labels
        :return: None
        """
        # get graphviz representation
        g = self.convert_graphviz(display_id)
        # write the .png filefor st in s
        g.draw(path+'.png',format='png',prog='dot')

    def is_consistent(self):
        """
        Check if the source description is consistent.
        -- check consistency of the semantic model;
        -- remove isolated nodes;
        :return:
        """
        pass


    def get_ontology_class_nodes(self, ontol):
        """
        Return a list of class nodes defined in the ontology.
        :param ontol: rdflib ontology graph
        :return: list of instances SemanticType
        """
        namespaces = dict([(y.split("#")[0], x) for (x, y) in ontol.namespaces()])
        class_nodes = []
        for c in ontol.subjects(RDF.type, OWL.Class):
            pref, lab = c.toPython().split("#")
            class_lab = namespaces[pref] + ':' + lab
            new_st = SemanticType(class_uri=c.toPython(), class_label=class_lab)
            class_nodes.append(new_st)
        return class_nodes

    def get_ontology_data_nodes(self, ontol):
        """
        Return a list of data nodes defined in the ontology.
        :param ontol: rdflib ontology graph
        :return: list of instances SemanticType
        """
        namespaces = {(y.split("#")[0], x) for x, y in ontol.namespaces()}
        data_nodes = []
        for d in ontol.subjects(RDF.type, OWL.DatatypeProperty): # these are data properties
            d_pref, d_lab = d.toPython().split("#")
            data_lab = namespaces[d_pref] + ':' + d_lab
            for c in ontol.objects(d, RDFS.domain): # these are classes for which data property d exists
                c_pref, c_lab = c.toPython().split("#")
                class_lab = namespaces[c_pref] + ':' + c_lab
                new_st = SemanticType(class_uri=c.toPython(), class_label=class_lab
                                      , data_uri=d.toPython(), data_label=data_lab)
                data_nodes.append(new_st)
        return data_nodes

    def get_sm_data_nodes(self, namespaces):
        """
        Get a list of semantic types which correspond to data nodes in the semantic model.
        :param namespaces: dictionary of namespaces specified in ontologies
        :return: list of instances SemanticType
        """
        data_nodes = self.semantic_model.get_data_nodes() # list of tuples (class_label, data_label)

    def get_semantic_types(self):
        """
        Display all semantic types from the ontology and semantic model.
        A semantic type is a tuple: (class_node, _) or (class_node,data_property).
        :return: list
        """
        semantic_types = []
        # get semantic types from ontologies which correspond to class nodes
        for ontol in self.ontologies:
            namespaces = dict([(y,x) for (x,y) in ontol.namespaces()])
            # class_nodes
            semantic_types += self.get_ontology_class_nodes(ontol)
            semantic_types += self.get_ontology_data_nodes(ontol)
        return semantic_types



########################################################################################
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
    sources_dir = os.path.join(data_dir,'sources')

    business = os.path.join(sources_dir,"businessInfo.csv")

    ontol_paths = ["/home/rue009/Projects/DFAT/SchemaMapping/Transformations/tests/example/dataintegration_report_ontology.owl"]

    op = "/home/rue009/Projects/DFAT/SchemaMapping/Transformations/tests/example/dataintegration_report_ontology.owl"
    ontol = read_ontology(op)

    # rdflib_to_networkx_multidigraph(ontol.graph)

    # generate_initial_sds_dir(data_dir, sources_dir, ontol_paths)

    fname = os.path.join(data_dir,"source_descriptions","businessInfo.ssd")
    new_sd = SourceDescription(read_sd(fname))

    sem_types = new_sd.get_semantic_types()
    print(sem_types)

    # dot_dir = os.path.join(data_dir, "ssd_graphviz")
    # sd_dir = os.path.join(data_dir,"source_descriptions")
    #
    # for f in os.listdir(sd_dir):
    #     print("Reading "+f)
    #     new_sd = SourceDescription(read_sd(os.path.join(sd_dir,f)))
    #     new_sd.write_dot(os.path.join(dot_dir,f+'.dot'), display_id=True)


