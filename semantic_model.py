import networkx as nx
import os
import logging
import matplotlib.pyplot as plt
import pygraphviz as pgv
import json


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


class SemanticModel(object):
    """
    Attributes:
        graph: networkx graph representation of the semantic model
    """
    def __init__(self, model_dict):
        """
        Initialize semantic model as a networkx graph.
        :param model_dict: dictionary with nodes and links.
        """
        logging.info("Initializing networkx graph to represent the semantic model.")
        self.graph = nx.MultiDiGraph()

        for n in model_dict["nodes"]:
            self.graph.add_node(int(n["id"]), n)

        for l in model_dict["links"]:
            self.graph.add_edge(int(l["source"]), int(l["target"]),
                                attr_dict={"type": l["type"], "label": l["label"]})

    def convert_graphviz(self):
        """
        Convert the semantic model to graphviz.
        :return:
        """
        # empty graphviz
        g = pgv.AGraph(strict=False,
                       directed=True,
                       remincross='true',
                       overlap=False,
                       splines='true')
        for (n, n_dict) in self.graph.nodes(data=True):
            if n_dict['type'] == 'ClassNode':
                g.add_node(n, label=n_dict['label'],
                           color='white',
                           style="filled",
                           fillcolor='green', shape='ellipse')
            else:
                g.add_node(n, label=n_dict['label'],
                           shape='plaintext',
                           color='white',
                           style='filled',
                           fillcolor='lightgrey')

        # put semantic model into cluster
        g.add_subgraph(self.graph.nodes(data=False))

        for (start, end, e_dict) in self.graph.edges(data=True):
            if e_dict["type"] == "ObjectProperty":
                g.add_edge(start, end, label=e_dict['label']
                           , fontname='times-italic')
            else:
                g.add_edge(start, end, label=e_dict['label'],
                           style="dashed"
                           , fontname='times-italic')

        return g

    def write_dot(self, path):
        """
        Write the .dot file for the semantic model using graphviz.
        :param path: string which indicates the location to write the .dot file.
        :return:
        """
        g = self.convert_graphviz()

        g.write(path)

        # # circles for class nodes and squares for data nodes
        # class_nodes = [n for (n,n_dict) in self.graph.nodes(data=True)
        #                if n_dict["type"]=='ClassNode']
        # print(class_nodes)
        # data_nodes = [n for (n, n_dict) in self.graph.nodes(data=True)
        #                if n_dict["type"] == 'DataNode']
        # print(data_nodes)
        #
        # nx.draw_networkx_nodes(self.graph, pos,
        #                        nodelist=class_nodes, alpha=0.4,
        #                        node_color='w',
        #                        node_shape='o')
        # nx.draw_networkx_nodes(self.graph, pos,
        #                        nodelist=data_nodes, alpha=0.4,
        #                        node_color='g',
        #                        node_shape='s')
        #
        # nx.draw_networkx_edges(self.graph, pos)
        #
        # labels = dict([(n, n_dict["label"]) for (n,n_dict) in self.graph.nodes(data=True)])
        #
        # nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)
        #
        # # nx.draw(self.graph)
        # plt.show()


    def visualize(self):
        """
        Draw the semantic model.
        :return:
        """
        # graph layout
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph)
        except:
            pos = nx.spring_layout(self.graph, iterations=20)

        # circles for class nodes and squares for data nodes
        class_nodes = [n for (n,n_dict) in self.graph.nodes(data=True)
                       if n_dict["type"]=='ClassNode']
        print(class_nodes)
        data_nodes = [n for (n, n_dict) in self.graph.nodes(data=True)
                       if n_dict["type"] == 'DataNode']
        print(data_nodes)

        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=class_nodes, alpha=0.4,
                               node_color='w',
                               node_shape='o')
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=data_nodes, alpha=0.4,
                               node_color='g',
                               node_shape='s')

        nx.draw_networkx_edges(self.graph, pos)

        labels = dict([(n, n_dict["label"]) for (n,n_dict) in self.graph.nodes(data=True)])

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)

        # nx.draw(self.graph)
        plt.show()

        # convert to GraphJSON format and throw it to alchemy.js

    def __repr__(self):
        return "<SemanticModel with " + str(self.graph.number_of_nodes()) \
               + " nodes and " + str(self.graph.number_of_nodes()) + " edges as " + \
               self.graph.__repr__() + ">"

    def getSemanticTypes(self):
        """
        Construct list of semantic types from the semantic model.
        :return:
        """
        pass

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    sources_dir = os.path.join(data_dir, 'sources')

    business = os.path.join(sources_dir, "businessInfo.csv")

    ontol_paths = ["</home/rue009/Projects/DFAT/SchemaMapping/Transformations/tests/example/dataintegration_report_ontology.owl>"]

    fname = os.path.join(data_dir,"source_descriptions","businessInfo.ssd")
    new_sd = read_sd(fname)

    # print(json.dumps(new_sd,indent=4))

    new_sm = SemanticModel(new_sd["semanticModel"])
    print("Model created")
    print(new_sm)

    # new_sm.visualize()

    new_sm.write_dot(fname+".dot")