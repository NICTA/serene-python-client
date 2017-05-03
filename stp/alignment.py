"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Converts serene intermediary output (alignment graph and matches) to construct an integration graph
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import networkx as nx
import json
import pandas as pd
from collections import defaultdict

try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene


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


def read_karma_graph(file_name, out_file):
    """
    convert graph.json from Karma to nx object
    :param file_name:
    :return:
    """
    logging.info("Reading karma alignment graph...")
    with open(file_name) as f:
        data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]

    g = nx.MultiDiGraph()

    cur_id = 0
    node_map = {}
    for node in nodes:
        # filling in the nodes
        node_map[node["id"]] = cur_id
        node_data = {
            "type": "ClassNode" if node["type"] == "InternalNode" else "DataNode",
            "label": label(node["label"]["uri"]) if node["type"] == "InternalNode" else "",
            "lab": label(node["id"]) if node["type"] == "InternalNode" else "",
            "prefix": prefix(node["label"]["uri"]) if node["type"] == "InternalNode" else ""
        }
        g.add_node(cur_id, attr_dict=node_data)
        cur_id += 1

    cur_id = 0
    for link in links:
        # filling in the links
        source, uri, target = link["id"].split("---")
        link_data={
            "type": link["type"],
            "weight": link["weight"],
            "label": label(uri),
            "prefix": prefix(uri)
        }
        g.add_edge(node_map[source], node_map[target], attr_dict=link_data)
        cur_id += 1
        if link["type"] in ["DataPropertyLink", "ClassInstanceLink"]:
            # change data nodes
            g.node[node_map[target]]["label"] = g.node[node_map[source]]["label"] + "---" + link_data["label"]
            g.node[node_map[target]]["lab"] = g.node[node_map[source]]["lab"] + "---" + link_data["label"]

    logging.info("Writing alignment graph to file {}".format(out_file))
    nx.write_graphml(g, out_file)

    logging.info("Karma alignment graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))

    return g


def add_matches(alignment_graph, predicted_df):
    """
    # add matches
    :param alignment_graph: networkx MultiDiGraph()
    :param predicted_df: pandas dataframe
    :return:
    """
    logging.info("Adding match nodes to the alignment graph...")

    # lookup from data node labels to node ids
    data_node_map = defaultdict(list)
    for n_id, n_data in alignment_graph.nodes(data=True):
        if n_data["type"] == "DataNode":
            data_node_map[n_data["label"]].append(n_id)

    available_types = set(data_node_map.keys())

    logging.info("Node map constructed: {}".format(data_node_map))
    # print(data_node_map)

    cur_id = 1 + alignment_graph.number_of_nodes() # we start nodes here from this number
    scores_cols = [col for col in predicted_df.columns if col.startswith("scores_")]

    for idx, row in predicted_df.iterrows():
        node_data = {
            "type": "Attribute",
            "label": row["column_name"]
        }
        alignment_graph.add_node(cur_id, attr_dict=node_data)
        for score in scores_cols:
            if row[score] == 0:
                logging.info("Ignoring 0 scores")
                continue
            lab = score.split("scores_")[1]
            # TODO: make weight: 1 - ln(score)
            link_data = {
                "type": "MatchLink",
                "weight": 1.0 - row[score], # inverse number
                "label": lab
            }
            if lab not in available_types:
                print("Weird label ", lab)
                logging.warning("Weird label {} for column".format(lab, row["column_name"]))
                continue
            for target in data_node_map[lab]:
                alignment_graph.add_edge(cur_id, target, attr_dict=link_data)

        cur_id += 1

    logging.info("Integration graph read: {} nodes, {} links".format(
        alignment_graph.number_of_nodes(), alignment_graph.number_of_edges()))

    return alignment_graph


def add_matches_file(alignment_graph, match_file):
    """
    # add matches
    :param alignment_graph:
    :param match_file:
    :return:
    """
    pass


def convert_ssd(ssd, integration_graph, file_name):
    """
    convert our ssd to nx object
    :param ssd:
    :param file_name:
    :return:
    """
    g = nx.MultiDiGraph()
    ssd.semantic_model._graph.nodes(data=True)



# write nx object to file


if __name__ == "__main__":

    # setting up the logging
    log_file = os.path.join("resources",'benchmark_stp.log')
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s: %(message)s',
                                      '%Y-%m-%d %H:%M:%S')
    my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=5, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    # logging.basicConfig(filename=log_file,
    #                     level=logging.DEBUG, filemode='w+',
    #                     format='%(asctime)s %(levelname)s %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(my_handler)

    karma_graph_file = os.path.join("resources", "1", "graph.json")
    # read alignment part
    alignment = read_karma_graph(karma_graph_file, os.path.join("resources", "1", "alignment.graphml"))

    # read matches
    predicted_df = pd.read_csv(os.path.join("resources", "1", "s02-dma.csv.csv.matches.csv"))
    # print(predicted_df)

    integration = add_matches(alignment, predicted_df)
    out_file = os.path.join("resources", "1", "integration.graphml")
    logging.info("Writing alignment graph to file {}".format(out_file))
    nx.write_graphml(integration, out_file)
