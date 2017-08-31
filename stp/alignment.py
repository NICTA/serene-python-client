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
from copy import deepcopy

try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene
from serene import Column, DataNode, ClassNode, ClassInstanceLink, ColumnLink, ObjectLink, DataLink,\
    ALL_CN, DEFAULT_NS, UNKNOWN_CN, SSD


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

    # maximum id for nodes which is currently in the lookup table
    max_node_id = max(max(v) for v in data_node_map.values())

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
                if len(node_id) < 1:
                    logging.warning("Label is absent in the integration graph {}".format(label))
                    print("Label is absent in the integration graph {}".format(label))
                    node_id = max_node_id + 1
                    max_node_id = node_id
                    data_node_map[label].append(node_id)
                else:
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
    if links:
        for l in links.values():
            if l["label"] == label:
                return l["weight"]
        return 0
    # link is not in the integration graph!
    return -1


def process_unknown(ssd, semantic_types):
    """
    Modify provided ssd in such way that semantic types which were absent in the training set are mapped
    to unknown.
    :param ssd:
    :param semantic_types:
    :return:
    """
    mod_ssd = deepcopy(ssd)
    logging.info("Modifying ssd by mapping semantic types which are absent in the train set to unknown")

    # semantic types present in the ssd
    ssd_labels = set(d.full_label for d in ssd.data_nodes)
    # semantic types which are not available in the train set
    absent_labels = ssd_labels.difference(set(semantic_types))
    # get those data nodes which are not in the semantic types
    absent_dn = [dn for dn in ssd.data_nodes if dn.full_label in absent_labels]

    # remove these data nodes
    logging.info("Removing {} data nodes".format(len(absent_dn)))
    for dn in absent_dn:
        mod_ssd.remove(dn)

    # map unmapped columns to the unknown class!
    mod_ssd.fill_unknown()

    # TODO: clean ssd: remove class nodes which have no data nodes or outgoing links!

    return mod_ssd


def convert_ssd(ssd, integration_graph, file_name=None, clean_ssd=True):
    """
    Convert our ssd to nx object re-using ids from the integration graph whenever possible
    :param ssd: SSD instance
    :param integration_graph: nx MultiDigraph
    :param file_name: name of the file to write the converted ssd; default None
    :param clean_ssd: boolean whether to remove extra "connect" links for the node "All"
    :return:
    """
    data_node_map = construct_data_node_map(integration_graph)
    logging.debug("data_node_map: {}".format(data_node_map))

    g, ssd_node_map, class_nodes = add_class_column(ssd, data_node_map)

    # maximum id for nodes which is currently in the lookup table
    max_node_id = max(max(v) for v in data_node_map.values())

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
            if full_data_label in data_node_map and\
                            len(data_node_map[full_data_label]) > len(data_nodes[full_data_label]):
                node_id = data_node_map[full_data_label][len(data_nodes[full_data_label])]
            else:
                logging.warning("Label {} is not present in the integration graph.".format(full_data_label))
                node_id = max_node_id + 1
                max_node_id = node_id
                data_node_map[full_data_label].append(node_id)
            data_nodes[full_data_label].append(node_id)

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

    # clean ssd: leave only 2 links for All
    if clean_ssd:
        all_id = [n for (n, n_data) in g.nodes(data=True) if
                  n_data["type"] == "ClassNode" and n_data["label"] == ALL_CN]
        unknown_id = [n for (n, n_data) in g.nodes(data=True) if
                  n_data["type"] == "ClassNode" and n_data["label"] == UNKNOWN_CN]
        if len(all_id) and len(unknown_id):
            all_id, unknown_id = all_id[0], unknown_id[0]
            all_neigh = [(s, t, ld["weight"]) for (s, t, ld) in g.out_edges([all_id], data=True) if t != unknown_id]
            min_weight = min(ld for (s, t, ld) in all_neigh)
            keep = min(t for (s, t, ld) in all_neigh if ld == min_weight)
            logging.info("Removing extra connect links: {}".format(len(all_neigh)-1))
            for s, t, ld in all_neigh:
                if t != keep:
                    g.remove_edge(s, t)

    print("SSD graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
    logging.info("SSD graph read: {} nodes, {} links".format(g.number_of_nodes(), g.number_of_edges()))
    if file_name:
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


#####################
#
#
####################

import time
import stp.integration as integ
# try:
#     import integration as integ
# except ImportError as e:
#     import sys
#     sys.path.insert(0, '.')
#     import stp.integration as integ

def create_octopus(server, ssd_range, sample, ontologies):
    octo_local = server.Octopus(
        ssds=[ssd_range[i] for i in sample],
        ontologies=ontologies,
        name='num_{}'.format(len(sample)),
        description='Testing example for places and companies',
        resampling_strategy="NoResampling",  # optional
        num_bags=80,  # optional
        bag_size=50,  # optional
        model_type="randomForest",
        modeling_props={
            "compatibleProperties": True,
            "ontologyAlignment": True,
            "addOntologyPaths": True,
            "mappingBranchingFactor": 50,
            "numCandidateMappings": 10,
            "topkSteinerTrees": 50,
            "multipleSameProperty": True,
            "confidenceWeight": 1.0,
            "coherenceWeight": 1.0,
            "sizeWeight": 1.0,
            "numSemanticTypes": 4,
            "thingNode": False,
            "nodeClosure": True,
            "propertiesDirect": True,
            "propertiesIndirect": True,
            "propertiesSubclass": True,
            "propertiesWithOnlyDomain": True,
            "propertiesWithOnlyRange": True,
            "propertiesWithoutDomainRange": False,
            "unknownThreshold": 0.05
        },
        feature_config={
            "activeFeatures": [
                "num-unique-vals",
                "prop-unique-vals",
                "prop-missing-vals",
                "ratio-alpha-chars",
                "prop-numerical-chars",
                "prop-whitespace-chars",
                "prop-entries-with-at-sign",
                "prop-entries-with-hyphen",
                "prop-range-format",
                "is-discrete",
                "entropy-for-discrete-values",
                "shannon-entropy"
            ]
            ,
            "activeFeatureGroups": [
                "char-dist-features",
                "inferred-data-type",
                "stats-of-text-length",
                "stats-of-numeric-type"
                , "prop-instances-per-class-in-knearestneighbours"
                , "mean-character-cosine-similarity-from-class-examples"
                , "min-editdistance-from-class-examples"
                , "min-wordnet-jcn-distance-from-class-examples"
                , "min-wordnet-lin-distance-from-class-examples"
            ],
            "featureExtractorParams": [
                {
                    "name": "prop-instances-per-class-in-knearestneighbours",
                    "num-neighbours": 5
                }
                , {
                    "name": "min-editdistance-from-class-examples",
                    "max-comparisons-per-class": 5
                }, {
                    "name": "min-wordnet-jcn-distance-from-class-examples",
                    "max-comparisons-per-class": 5
                }, {
                    "name": "min-wordnet-lin-distance-from-class-examples",
                    "max-comparisons-per-class": 5
                }
            ]
        }
    )

    # add this to the endpoint...
    print("Now we upload to the server")
    octo = server.octopii.upload(octo_local)

    start = time.time()
    print()
    print("     the initial state for {} is {}".format(octo.id, octo.state))
    print("     training...")
    octo.train()
    print("     Done in: {}".format(time.time() - start))
    print("     the final state for {} is {}".format(octo.id, octo.state))

    return octo


def create_octopus2(server, ssd_range, sample, ontologies):
    octo_local = server.Octopus(
        ssds=[ssd_range[i] for i in sample],
        ontologies=ontologies,
        name='num_{}'.format(len(sample)),
        description='Testing example for places and companies',
        resampling_strategy="NoResampling",  # optional
        num_bags=80,  # optional
        bag_size=50,  # optional
        model_type="randomForest",
        modeling_props={
            "compatibleProperties": True,
            "ontologyAlignment": True,
            "addOntologyPaths": False,
            "mappingBranchingFactor": 50,
            "numCandidateMappings": 10,
            "topkSteinerTrees": 50,
            "multipleSameProperty": True,
            "confidenceWeight": 1.0,
            "coherenceWeight": 1.0,
            "sizeWeight": 1.0,
            "numSemanticTypes": 4,
            "thingNode": False,
            "nodeClosure": True,
            "propertiesDirect": True,
            "propertiesIndirect": True,
            "propertiesSubclass": True,
            "propertiesWithOnlyDomain": True,
            "propertiesWithOnlyRange": True,
            "propertiesWithoutDomainRange": False,
            "unknownThreshold": 0.05
        },
        feature_config={
            "activeFeatures": [
                "num-unique-vals",
                "prop-unique-vals",
                "prop-missing-vals",
                "ratio-alpha-chars",
                "prop-numerical-chars",
                "prop-whitespace-chars",
                "prop-entries-with-at-sign",
                "prop-entries-with-hyphen",
                "prop-range-format",
                "is-discrete",
                "entropy-for-discrete-values",
                "shannon-entropy"
            ]
            ,
            "activeFeatureGroups": [
                "char-dist-features",
                "inferred-data-type",
                "stats-of-text-length",
                "stats-of-numeric-type"
                , "prop-instances-per-class-in-knearestneighbours"
                , "mean-character-cosine-similarity-from-class-examples"
                , "min-editdistance-from-class-examples"
                , "min-wordnet-jcn-distance-from-class-examples"
                , "min-wordnet-lin-distance-from-class-examples"
            ],
            "featureExtractorParams": [
                {
                    "name": "prop-instances-per-class-in-knearestneighbours",
                    "num-neighbours": 3
                }
                , {
                    "name": "min-editdistance-from-class-examples",
                    "max-comparisons-per-class": 3
                }, {
                    "name": "min-wordnet-jcn-distance-from-class-examples",
                    "max-comparisons-per-class": 3
                }, {
                    "name": "min-wordnet-lin-distance-from-class-examples",
                    "max-comparisons-per-class": 3
                }
            ]
        }
    )
    # add this to the endpoint...
    print("Now we upload to the server")
    octo = server.octopii.upload(octo_local)

    start = time.time()
    print()
    print("     the initial state for {} is {}".format(octo.id, octo.state))
    print("     training...")
    octo.train()
    print("     Done in: {}".format(time.time() - start))
    print("     the final state for {} is {}".format(octo.id, octo.state))

    return octo


def create_octopus3(server, ssd_range, sample, ontologies):
    octo_local = server.Octopus(
        ssds=[ssd_range[i] for i in sample],
        ontologies=ontologies,
        name='num_{}'.format(len(sample)),
        description='Testing example for places and companies',
        resampling_strategy="NoResampling",  # optional
        num_bags=80,  # optional
        bag_size=50,  # optional
        model_type="randomForest",
        modeling_props={
            "compatibleProperties": True,
            "ontologyAlignment": True,
            "addOntologyPaths": False,
            "mappingBranchingFactor": 50,
            "numCandidateMappings": 10,
            "topkSteinerTrees": 50,
            "multipleSameProperty": False,
            "confidenceWeight": 1.0,
            "coherenceWeight": 1.0,
            "sizeWeight": 1.0,
            "numSemanticTypes": 4,
            "thingNode": False,
            "nodeClosure": True,
            "propertiesDirect": True,
            "propertiesIndirect": True,
            "propertiesSubclass": True,
            "propertiesWithOnlyDomain": True,
            "propertiesWithOnlyRange": True,
            "propertiesWithoutDomainRange": False,
            "unknownThreshold": 0.05
        },
        feature_config={
            "activeFeatures": [
                "num-unique-vals",
                "prop-unique-vals",
                "prop-missing-vals",
                "ratio-alpha-chars",
                "prop-numerical-chars",
                "prop-whitespace-chars",
                "prop-entries-with-at-sign",
                "prop-entries-with-hyphen",
                "prop-range-format",
                "is-discrete",
                "entropy-for-discrete-values",
                "shannon-entropy"
            ]
            ,
            "activeFeatureGroups": [
                "char-dist-features",
                "inferred-data-type",
                "stats-of-text-length",
                "stats-of-numeric-type"
                , "prop-instances-per-class-in-knearestneighbours"
                , "mean-character-cosine-similarity-from-class-examples"
                , "min-editdistance-from-class-examples"
                , "min-wordnet-jcn-distance-from-class-examples"
                , "min-wordnet-lin-distance-from-class-examples"
            ],
            "featureExtractorParams": [
                {
                    "name": "prop-instances-per-class-in-knearestneighbours",
                    "num-neighbours": 3
                }
                , {
                    "name": "min-editdistance-from-class-examples",
                    "max-comparisons-per-class": 3
                }, {
                    "name": "min-wordnet-jcn-distance-from-class-examples",
                    "max-comparisons-per-class": 3
                }, {
                    "name": "min-wordnet-lin-distance-from-class-examples",
                    "max-comparisons-per-class": 3
                }
            ]
        }
    )
    # add this to the endpoint...
    print("Now we upload to the server")
    octo = server.octopii.upload(octo_local)

    start = time.time()
    print()
    print("     the initial state for {} is {}".format(octo.id, octo.state))
    print("     training...")
    octo.train()
    print("     Done in: {}".format(time.time() - start))
    print("     the final state for {} is {}".format(octo.id, octo.state))

    return octo



def semantic_label_df(cur_ssd, col_name="pred_label"):
    """
    Get schema matcher part from ssd
    :param cur_ssd: SSD
    :param col_name: name to be given to the column which has the label
    :return: pandas dataframe
    """
    ff = [(v.id, k.class_node.label + "---" + k.label) for (k, v) in cur_ssd.mappings.items()]
    return pd.DataFrame(ff, columns=["column_id", col_name])


import sklearn.metrics
def compare_semantic_label(one_df, two_df):
    """

    :param one_df:
    :param two_df:
    :return:
    """
    logging.info("Comparing semantic labels")
    merged = one_df.merge(two_df, on="column_id", how="outer")
    y_true = merged["user_label"].astype(str).as_matrix()
    y_pred = merged["label"].astype(str).as_matrix()
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)  # np.mean(y_pred == y_true)
    fmeasure = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    return {"accuracy": accuracy, "fmeasure": fmeasure}

def do_chuffed(server, ch_octopus, ch_dataset, ch_column_names, ch_ssd, orig_ssd,
               train_flag=False, chuffed_path=None, simplify=True,
               patterns=None, experiment="loo", soft_assumptions=False,
               pattern_sign=1.0, accu=0.0, pattern_time=None):
    res = []
    start = time.time()
    if patterns is None:
        pattern_time = None
    try:
        integrat = integ.IntegrationGraph(octopus=ch_octopus, dataset=ch_dataset,
                                          simplify_graph=simplify,
                                          patterns=patterns, pattern_sign=pattern_sign,
                                          add_unknown=True)
        if chuffed_path:
            integrat._chuffed_path = chuffed_path  # change chuffed solver version
        solution = integrat.solve(ch_column_names,
                                  soft_assumptions=soft_assumptions, relax_every=2)
        runtime = time.time() - start
        # get cost of the ground truth
        try:
            nx_ssd = convert_ssd(ch_ssd, integrat.graph, file_name=None, clean_ssd=False)
            cor_match_score = sum([ld["weight"] for n1, n2, ld in nx_ssd.edges(data=True) if ld["type"] == "MatchLink"])
            cor_cost = sum([ld["weight"] for n1, n2, ld in nx_ssd.edges(data=True) if ld["type"] != "MatchLink"])
        except Exception as e:
            logging.warning("Failed to convert correct ssd with regard to the integration graph: {}".format(e))
            print("Failed to convert correct ssd with regard to the integration graph: {}".format(e))
            cor_match_score = None
            cor_cost=None

        if len(solution) < 1:
            logging.error("Chuffed found no solutions for ssd {}".format(orig_ssd.id))
            raise Exception("Chuffed found no solutions!")
        max_sol = len(solution) -1

        for idx, sol in enumerate(solution):
            # convert solution to SSD
            json_sol = json.loads(sol)
            logging.info("Converted solution: {}".format(json_sol))
            cp_ssd = SSD().update(json_sol, server._datasets, server._ontologies)
            eval = cp_ssd.evaluate(ch_ssd, False, True)

            # time returned by chuffed
            chuffed_time = json_sol["time"]
            # match score calculated by chuffed
            match_score = json_sol["match_score"]
            # cost of the calculated steiner tree
            cost = json_sol["cost"]
            # value of the calculated objective function
            objective = json_sol["objective"]
            extended_time = json_sol["extended_time"]
            # optimal soluiotn is the last found by chuffed
            optimal_flag = json_sol["optimalFlag"] & idx == max_sol

            try:
                comparison = server.ssds.compare(cp_ssd, ch_ssd, False, False)
            except:
                comparison = {"jaccard": -1, "precision": -1, "recall": -1}

            # comparison = {"jaccard": -1, "precision": -1, "recall": -1}
            try:
                sol_accuracy = compare_semantic_label(semantic_label_df(ch_ssd, "user_label"),
                                                  semantic_label_df(cp_ssd, "label"))["accuracy"]
            except Exception as e:
                logging.error("Match score fail in chuffed: {}".format(e))
                print("Match score fail in chuffed: {}".format(e))
                sol_accuracy = None

            res += [[experiment, ch_octopus.id, ch_dataset.id, orig_ssd.name,
                     orig_ssd.id, "chuffed", idx, "success",
                     eval["jaccard"], eval["precision"], eval["recall"],
                     comparison["jaccard"], comparison["precision"], comparison["recall"],
                     train_flag, runtime, chuffed_path, simplify, chuffed_time,
                     soft_assumptions, pattern_sign, accu, sol_accuracy,
                     match_score, cost, objective,
                     cor_match_score, cor_cost, pattern_time,
                     integrat.graph.number_of_nodes(), integrat.graph.number_of_edges(), extended_time, optimal_flag]]
    except Exception as e:
        print("CP solver failed to find solution: {}".format(e))
        res += [[experiment, ch_octopus.id, ch_dataset.id, orig_ssd.name, orig_ssd.id,
                 "chuffed", 0, "fail", None, None, None, None, None, None,
                 train_flag, time.time() - start, chuffed_path, simplify, None,
                 soft_assumptions, pattern_sign, accu, None,
                 None, None, None, None, None, pattern_time, None, None, None, None]]
    return res

def do_karma(server, k_octopus, k_dataset, k_ssd, orig_ssd,
             train_flag=False, experiment="loo", accu=0.0, pattern_time=None):
    start = time.time()
    try:
        karma_pred = k_octopus.predict(k_dataset)[0]
        karma_ssd = karma_pred.ssd
        runtime = time.time() - start
        eval = karma_ssd.evaluate(k_ssd, False, True)

        match_score = karma_pred.score.nodeConfidence
        cost = karma_pred.score.linkCost
        objective = karma_pred.score.karmaScore

        try:
            comparison = server.ssds.compare(karma_ssd, k_ssd, False, False)
        except:
            comparison = {"jaccard": -1, "precision": -1, "recall": -1}
        # comparison = {"jaccard": -1, "precision": -1, "recall": -1}
        try:
            sol_accuracy = compare_semantic_label(semantic_label_df(k_ssd, "user_label"),
                                              semantic_label_df(karma_ssd, "label"))["accuracy"]
        except Exception as e:
            logging.error("Match accuracy fail: {}".format(e))
            print("Match accuracy fail: {}".format(e))
            sol_accuracy = None

        return [[experiment, k_octopus.id, k_dataset.id, orig_ssd.name,
                 orig_ssd.id, "karma", 0, "success",
                 eval["jaccard"], eval["precision"], eval["recall"],
                 comparison["jaccard"], comparison["precision"], comparison["recall"],
                 train_flag, runtime, "", None, runtime, None, None, accu, sol_accuracy,
                 match_score, cost, objective, None, None, None, None, None, None, None]]
    except Exception as e:
        logging.error("Karma failed to find solution: {}".format(e))
        print("Karma failed to find solution: {}".format(e))
        return [[experiment, k_octopus.id, k_dataset.id, orig_ssd.name,
                 orig_ssd.id, "karma", 0, "fail", None, None, None, None, None, None,
                train_flag, time.time() - start, "", None, None, None, None, accu, None,
                 None, None, None, None, None, None, None, None, None, None]]


#####################


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


