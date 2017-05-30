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
import pygraphviz as pgv

try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene
from serene import Column, DataNode, ClassNode, ClassInstanceLink, ColumnLink, ObjectLink, DataLink


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


def read_karma_graph(file_name, out_file=None):
    """
    convert graph.json from Karma to nx object
    :param file_name:
    :param out_file: Option string to write down the graph
    :return:
    """
    logging.info("Reading karma alignment graph...")
    with open(file_name) as f:
        data = json.load(f)

    g = convert_karma_graph(data)

    if out_file:
        logging.info("Writing alignment graph to file {}".format(out_file))
        nx.write_graphml(g, out_file)

    return g


def convert_karma_graph(data):
    """
    convert graph.json from Karma to nx object
    :param data: json read dictionary
    :return:
    """
    logging.info("Converting karma alignment graph...")

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
            g.node[node_map[target]]["prefix"] = g.node[node_map[source]]["prefix"]

    logging.info("Karma alignment graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
    print("Karma alignment graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))

    return g


def clean_alignment(alignment_graph):
    """
    Remove extra links for the unknown node
    :param alignment_graph:
    :return:
    """
    unknown_nodes = []
    for n_id, nd in alignment_graph.nodes(data=True):
        if nd["label"] == "Unknown":
            unknown_nodes.append(n_id)
    if len(unknown_nodes):
        print("Cleaning unknown nodes: {}".format(len(unknown_nodes)))
        for unknown_node in unknown_nodes:
            links = alignment_graph.edges([unknown_node], data=True)
            for l_start, l_end, ld in links:
                if ld["type"] not in {'DataPropertyLink', 'SubClassLink', 'ObjectPropertyLink'}:
                    alignment_graph.remove_edge(l_start, l_end)

    return alignment_graph

def construct_data_node_map(integration_graph):
    """
    Construct lookup table for node labels to node ids in the integration graph.
    :param integration_graph:
    :return:
    """
    data_node_map = defaultdict(list)
    for n_id, n_data in integration_graph.nodes(data=True):
        if n_data["type"] == "DataNode":
            prefix = n_data["prefix"]
            data_node_map[prefix + n_data["lab"]].append(n_id)
        elif n_data["type"] == "ClassNode":
            prefix = n_data["prefix"]
            data_node_map[prefix + n_data["lab"]].append(n_id)
        elif n_data["type"] == "Attribute":
            data_node_map[n_data["label"]].append(n_id)
    return data_node_map


def add_matches(alignment_graph, predicted_df, column_names, threshold=0):
    """
    add matches to the alignment graph
    :param alignment_graph: networkx MultiDiGraph()
    :param predicted_df: pandas dataframe
    :param column_names: list of column names which are present in the ssd
    :param threshold: discard matches with score < threshold
    :return:
    """
    logging.info("Adding match nodes to the alignment graph...")
    threshold = min(max(0.0, threshold),1.0) # threshold should be in range [0,1]
    new_graph = alignment_graph.copy()
    # lookup from data node labels to node ids
    data_node_map = defaultdict(list)
    for n_id, n_data in alignment_graph.nodes(data=True):
        if n_data["type"] == "DataNode":
            data_node_map[n_data["label"]].append(n_id)

    # we add matches only for those attributes which are mapped to existing data nodes
    available_types = set(data_node_map.keys())

    logging.info("Node map constructed: {}".format(data_node_map))
    # print(data_node_map)

    cur_id = 1 + new_graph.number_of_nodes()  # we start nodes here from this number
    scores_cols = [col for col in predicted_df.columns if col.startswith("scores_")]

    for idx, row in predicted_df.iterrows():
        if row["column_name"] not in column_names:
            continue
        node_data = {
            "type": "Attribute",
            "label": row["column_name"]
        }
        new_graph.add_node(cur_id, attr_dict=node_data)
        for score in scores_cols:
            if row[score] <= threshold:
                logging.info("Ignoring <=threshold scores")
                continue
            lab = score.split("scores_")[1]
            # TODO: make weight: 1 - ln(score)
            link_data = {
                "type": "MatchLink",
                "weight": 1.0 - row[score],  # inverse number
                "label": row["column_name"]
            }
            if lab not in available_types:
                logging.warning("Weird label {} for column".format(lab, row["column_name"]))
                continue
            for target in data_node_map[lab]:
                new_graph.add_edge(target, cur_id, attr_dict=link_data)

        cur_id += 1

    print("Integration graph read: {} nodes, {} links".format(
        new_graph.number_of_nodes(), new_graph.number_of_edges()))
    logging.info("Integration graph read: {} nodes, {} links".format(
        new_graph.number_of_nodes(), new_graph.number_of_edges()))

    return new_graph


def add_class_column(ssd, data_node_map):
    """
    Add class nodes and attributes to the converted ssd
    :param g:
    :param ssd:
    :param class_nodes:
    :param data_node_map:
    :return:
    """
    logging.info("Adding class nodes and attributes")
    g = nx.MultiDiGraph()
    ssd_node_map = {}
    class_nodes = defaultdict(list)  # keeping track how many instances of the same class are present
    data_nodes = defaultdict(list)  # keeping track how many instances of the same data property are present
    for n_id, n_data in ssd.semantic_model._graph.nodes(data=True):
        data = n_data["data"]
        if isinstance(data, ClassNode):
            uri = data.prefix + data.label
            if uri not in class_nodes:
                label = uri + "1"
                lab = data.label + "1"
                node_id = data_node_map[label]
                if len(node_id) > 1:
                    logging.warning("More than one node per class {}".format(label))
                node_id = node_id[0]
                class_nodes[uri].append(node_id)
            else:
                label = uri + str(len(class_nodes[uri]) + 1)
                lab = data.label + str(len(class_nodes[uri]) + 1)
                node_id = data_node_map[label]
                if len(node_id) > 1:
                    logging.warning("More than one node per class {}".format(label))
                node_id = node_id[0]
                class_nodes[uri].append(node_id)
            node_data = {
                "type": "ClassNode",
                "label": data.label,
                "lab": lab,
                "prefix": data.prefix
            }
            ssd_node_map[n_id] = node_id
        elif isinstance(data, Column):
            node_id = data_node_map[data.name][len(data_nodes[data.name])]
            data_nodes[data.name].append(node_id)
            node_data = {
                "type": "Attribute",
                "label": data.name,
                "prefix": ""
            }
            ssd_node_map[n_id] = node_id
        else:
            continue
        g.add_node(node_id, attr_dict=node_data)

    return g, ssd_node_map, class_nodes


def get_weight(source, target, label, integration_graph):
    """
    Return the weight of the link between source and target with the specified label
    :param source:
    :param target:
    :param label:
    :param integration_graph:
    :return:
    """
    links = integration_graph.get_edge_data(source, target)
    for l in links.values():
        if l["label"] == label:
            return l["weight"]
    return 0


def convert_ssd(ssd, integration_graph, file_name):
    """
    Convert our ssd to nx object
    :param ssd:
    :param integration_graph:
    :param file_name:
    :return:
    """
    data_node_map = construct_data_node_map(integration_graph)
    logging.debug("data_node_map: {}".format(data_node_map))

    g, ssd_node_map, class_nodes = add_class_column(ssd, data_node_map)

    print("ssd_node_map: ")
    data_nodes = defaultdict(list)  # keeping track how many instances of the same data property are present
    for source, target, l_data in ssd.semantic_model._graph.edges(data=True):
        data = l_data["data"]
        if isinstance(data, ObjectLink):

            link_data = {
                "type": "ObjectPropertyLink",
                "label": data.label,
                "prefix": data.prefix,
                "weight": get_weight(ssd_node_map[source], ssd_node_map[target], data.label, integration_graph)
            }
            g.add_edge(ssd_node_map[source], ssd_node_map[target], attr_dict=link_data)
        elif isinstance(data, DataLink) or isinstance(data, ClassInstanceLink):
            # we need to add the data node
            class_node = g.node[ssd_node_map[source]]
            data_node = ssd.semantic_model._graph.node[target]["data"]
            link_label = data.label
            full_data_label = class_node["prefix"] + class_node["lab"] + "---" + link_label
            node_id = data_node_map[full_data_label][len(data_nodes[full_data_label])]
            data_nodes[full_data_label].append(full_data_label)
            # if len(node_id) != 1:
            #     logging.warning("DataNode {} is not well present in th integration graph: {}".format(
            #         full_data_label, len(node_id)))
            #     print("DataNode {} is not well present in the integration graph: {}".format(
            #         full_data_label, len(node_id)))
            # node_id = node_id[0]
            node_data = {
                "type": "DataNode",
                "label": class_node["label"] + "---" + link_label,
                "lab": class_node["lab"] + "---" + link_label,
                "prefix": class_node["prefix"]
            }
            g.add_node(node_id, attr_dict=node_data)
            ssd_node_map[target] = node_id
            logging.debug("---> Adding target {} with node {}".format(target, data_node))
            link_data = {
                "type": "DataPropertyLink" if isinstance(data, DataLink) else "ClassInstanceLink",
                "label": data.label,
                "prefix": data.prefix,
                "weight": get_weight(ssd_node_map[source], node_id, data.label, integration_graph)
            }
            g.add_edge(ssd_node_map[source], node_id, attr_dict=link_data)
        else:
            continue

    logging.debug("ssd_node_map: {}".format(ssd_node_map))
    for source, target, l_data in ssd.semantic_model._graph.edges(data=True):
        data = l_data["data"]
        if isinstance(data, ColumnLink):
            link_data = {
                "type": "MatchLink",
                "label": data.label,
                "weight": get_weight(ssd_node_map[source], ssd_node_map[target], data.label, integration_graph)
            }
            g.add_edge(ssd_node_map[source], ssd_node_map[target], attr_dict=link_data)

    # add data nodes
    # for n_id, n_data in ssd.semantic_model._graph.nodes(data=True):
    #     print("id={}, data={}".format(n_id, n_data))
    #     data = n_data["data"]
    #     # if isinstance(data, ClassNode):

    print("SSD graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
    logging.info("SSD graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
    logging.info("Writing converted ssd graph to file {}".format(file_name))
    nx.write_graphml(g, file_name)

    return g


def to_graphviz(graph, out_file=None, prog="dot"):
    """
    Convert networkx graph to graphviz object
    :param g: nx graph
    :param out_file: file name to write the dot
    :return:
    """

    g = pgv.AGraph(directed=True,
                   remincross='true',
                   fontname='helvetica',
                   overlap=False)
    styles = {}
    styles["DataNode"] = {"shape": "plaintext",
                       "style": "filled",
                       "fillcolor":"gold",
                       "fontname": "helvetica"
                       }
    styles["Attribute"] = {"shape": "hexagon",
                       "fontname": "helvetica"
                       }

    styles["ClassNode"] = {"shape": "ellipse",
                           "style": "filled",
                           "fillcolor":"#59d0a0",
                           "color": "white",
                           "fontname": "helvetica"
                           }
    styles["MatchLink"] = {"color": "brown", "fontcolor":"black", "fontname":"helvetice"}
    styles["Link"] = {"color": "#59d0a0", "fontcolor": "black", "fontname":"helvetice"}

    pgv_nodes = set()
    for source, target, l_data in graph.edges(data=True):
        if source not in pgv_nodes:
            n = graph.node[source]
            g.add_node(source, label=str(source)+"\n"+n["label"],
                       **styles[n["type"]])
            pgv_nodes.add(source)
        if target not in pgv_nodes:
            n = graph.node[target]
            g.add_node(target, label=str(target)+"\n"+n["label"],
                       **styles[graph.node[target]["type"]])
            pgv_nodes.add(target)

        edge_label = l_data["label"] + "\nw=%.3f" % l_data["weight"]
        if l_data["type"] == "MatchLink":
            edge_label = "w=%.3f" % l_data["weight"]
            g.add_edge(source, target,
                       label=edge_label, **styles["MatchLink"])
        else:
            g.add_edge(source, target,
                       label=edge_label, **styles["Link"])

    if out_file:
        print("writing graph dot file ", out_file)
        # g.layout()
        # g.draw(out_file, prog='dot')
        g.draw(out_file, prog=prog)
    return g


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

    karma_graph_file = os.path.join("resources", "test", "graph.json")
    # read alignment part
    alignment = read_karma_graph(karma_graph_file, os.path.join("resources", "test", "alignment.graphml"))

    # read matches
    predicted_df = pd.read_csv(os.path.join("resources", "test", "s02-dma.csv.csv.matches.csv"))
    # print(predicted_df)

    sn = serene.Serene(
        host='127.0.0.1',
        port=8080,
    )
    print(sn)

    for ssd in sn.ssds.items:
        if ssd.name == "s02-dma.csv":
            break
    print("---SSD", ssd)
    print("Num columns: ", len(ssd.columns))
    column_names = [c.name for c in ssd.columns]
    print("column names: ", column_names)
    print("predicted_df size: ", predicted_df.shape)

    integration = add_matches(alignment, predicted_df, column_names)
    out_file = os.path.join("resources", "test", "integration.graphml")
    logging.info("Writing integration graph to file {}".format(out_file))
    nx.write_graphml(integration, out_file)

    data_node_map = construct_data_node_map(integration)
    print("Data node map: ", data_node_map)


    out_file = os.path.join("resources", "test", ssd.name+".graphml")
    ssd_graph = convert_ssd(ssd, integration, out_file)

    print("Graphviz ssd")
    g = to_graphviz(ssd_graph, "ssd_graph.dot")
    print("Graphviz integration")
    g = to_graphviz(integration, "integration_graph.dot")


