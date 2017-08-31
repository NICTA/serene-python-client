"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Implementation of the class to map a relational data source onto an ontology using a CP solver,
in particular CHUFFED solver for constraint programming
"""

import logging
import networkx as nx
import yaml
import os
from collections import defaultdict

try:
    from minizinc import *
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    from stp.minizinc import *

import subprocess
import re
import time
import json
from pygraphviz import AGraph
from serene.elements.semantics.base import KARMA_DEFAULT_NS, DEFAULT_NS, ALL_CN, OBJ_PROP, UNKNOWN_CN, UNKNOWN_DN


class IntegrationGraph(object):
    """
    Integration graph to map a relational data source onto an ontology using a CP solver
    """
    UNKNOWN_SCORE = 5.0
    DEFAULT_SCORE = 99.0

    def __init__(self, octopus, dataset, match_threshold=0.0,
                 simplify_graph=False,
                 patterns=None,
                 pattern_sign=1.0,
                 predicted_df=None,
                 alignment_graph=None,
                 alignment_node_map=None,
                 alignment_link_map=None,
                 add_unknown=False):
        """
        Octopus should be trained.
        :param octopus: Octopus object which is used to integrate the new data source
        :param dataset: Dataset object for which integration is performed
        :param match_threshold: threshold on match links
        :param simplify_graph: Remove edges that are heavier than some path between its endpoints
        :param patterns: csv file with patterns, None if there will be no patterns
        :param pattern_sign: positive float number which indicates significance of pattern weights
                (they will be divided by this number)
        :param predicted_df: pandas dataframe with predictions
        :param alignment_graph: networkx multidigraph for the alignment graph
        :param alignment_node_map: dictionary
        :param alignment_link_map: dictionary
        :param add_unknown
        :return:
        """
        logging.info("Initializing integration graph for the dataset {} using octopus {}".format(
            dataset.id, octopus.id))
        self._path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stp")
        self._octopus = octopus
        self._dataset = dataset
        self._match_threshold = match_threshold  # discard matches with confidence score < threshold
        self._simplify_graph = simplify_graph
        self._patterns = patterns
        if pattern_sign <= 0:  # this number can be only positive
            pattern_sign = 1.0
        self._pattern_sign = pattern_sign
        self._predicted_df = predicted_df
        self._add_unknown = add_unknown

        # the integration graph contains the alignment graph at the beginning
        self.graph, self._node_map, self._link_map = self._get_initial_alignment(alignment_graph,
                                                                                 alignment_node_map,
                                                                                 alignment_link_map,
                                                                                 add_unknown)
        self._chuffed_path, self._mzn2fzn_exec, self._model_mzn, self._solns2out_exec, self._timeout = \
            self._read_config()

        self._graph_node_map = {}

    def _get_initial_alignment(self, alignment_graph,
                               alignment_node_map, alignment_link_map,
                               add_unknown=False):
        """
        Get initial alignment based on the octopus.
        Depending on the dataset, we do some further changes
        :param alignment_graph
        :param alignment_node_map
        :param alignment_link_map
        :param add_unknown
        :return:
        """

        if alignment_graph is not None and alignment_node_map is not None and alignment_link_map is not None:
            graph, node_map, link_map = alignment_graph, alignment_node_map, alignment_link_map
        else:
            graph, node_map, link_map = self._octopus.get_alignment()

        # modify the number of unknown data nodes in the alignment graph depending on the number of attributes
        num_attrs = len(self._dataset.columns)

        unknown_dn = [n_id for (n_id, n_data) in graph.nodes(data=True) if n_data["type"] == "DataNode" and
                      n_data["label"] == UNKNOWN_CN + "---" + UNKNOWN_DN]

        unknown_cn = [n_id for (n_id, n_data) in graph.nodes(data=True) if n_data["type"] == "ClassNode" and
                      n_data["label"] == UNKNOWN_CN]
        if len(unknown_cn) > 1:
            logging.error("Alignment graph has more than one unknown class node. This is wrong")
            raise Exception("Alignment graph has more than one unknown class node. This is wrong")
        if len(unknown_cn) < 1:
            if add_unknown:
                logging.info("Adding unknown class node")
                class_nodes = [n_id for (n_id, n_data) in graph.nodes(data=True) if n_data["type"] == "ClassNode"]
                # add more Unknown and All class nodes
                max_id = max(n for n in graph.nodes()) + 1
                node_data = {
                    "type": "ClassNode",
                    "label": UNKNOWN_CN,
                    "lab": UNKNOWN_CN + "1",
                    "prefix": DEFAULT_NS
                }
                graph.add_node(max_id, attr_dict=node_data)
                node_data = {
                    "type": "ClassNode",
                    "label": ALL_CN,
                    "lab": ALL_CN + "1",
                    "prefix": DEFAULT_NS
                }
                graph.add_node(max_id+1, attr_dict=node_data)
                # we need to add links
                link_data = {
                    "type": "ObjectPropertyLink",
                    "weight": self.DEFAULT_SCORE,
                    "label": OBJ_PROP,
                    "prefix": DEFAULT_NS,
                    'alignId': DEFAULT_NS + ALL_CN + "1---" +
                               DEFAULT_NS + OBJ_PROP + "---" +
                               DEFAULT_NS + UNKNOWN_CN + "1"
                }
                max_edge_id = max(k for (source, target, k) in graph.edges(data=False, keys=True)) + 1
                graph.add_edge(max_id+1, max_id, attr_dict=link_data)  # All --> Unknown
                for (n_id, inc) in enumerate(class_nodes):
                    link_data["alignId"] = DEFAULT_NS + ALL_CN + "1---" + \
                                           DEFAULT_NS + OBJ_PROP + "---" + \
                                           DEFAULT_NS + graph.node[n_id]["label"]
                    graph.add_edge(max_id + 1, n_id, key= max_edge_id + inc,
                                   attr_dict=link_data)  # All --> ClassNode

                unknown_cn = [max_id]

            else:
                logging.info("There is no unknown class node. "
                             "No manipulation to the original alignment graph will be done.")
                return graph, node_map, link_map

        if num_attrs > len(unknown_dn) and add_unknown:
            logging.info("Adding more unknown data nodes...")
            # add more unknown data nodes
            max_id = max(n for n in graph.nodes()) + 1
            unknown_cn = unknown_cn[0]
            max_edge = max(k for (source, target, k) in graph.edges(data=False, keys=True)) + 1

            unknown_data = {'lab': 'Unknown1---unknown', 'label': UNKNOWN_CN + "---" + UNKNOWN_DN,
                            'prefix': DEFAULT_NS,
                            'type': 'DataNode'}
            for i in range(num_attrs - len(unknown_dn)):
                graph.add_node(max_id + i, attr_dict=unknown_data)
                unknown_edata = {'alignId': DEFAULT_NS + UNKNOWN_CN + "1---" +
                                            DEFAULT_NS + UNKNOWN_DN + "---attr" + str(i),
                                 'label': UNKNOWN_DN,
                                 'prefix': DEFAULT_NS,
                                 'type': 'DataPropertyLink',
                                 'weight': self.UNKNOWN_SCORE}
                graph.add_edge(unknown_cn, max_id+i, key=max_edge+i, attr_dict=unknown_edata)

        # elif num_attrs < len(unknown_dn):
        #     # remove extra unknown data nodes
        #     logging.info("Removing extra unknown data nodes...")
        #     [graph.remove_node(n_id) for n_id in unknown_dn[num_attrs:]]

        return graph, node_map, link_map

    def _read_config(self):
        logging.info("Reading configuration parameters for the integration graph")

        with open(os.path.join(self._path, 'config.yaml'), 'r') as stream:
            config = yaml.load(stream)

        return config["chuffed-path"], config["mzn2fzn-exec"], \
               config["model-path"], config["solns2out-exec"], config["timeout"]

    def __str__(self):
        return "<IntegrationGraph({}, {}, " \
               "match_threshold={}, simplify={}, patterns={})>".format(self._octopus,
                                                                       self._dataset,
                                                                       self._match_threshold,
                                                                       self._simplify_graph,
                                                                       self._patterns)

    def __repr__(self):
        return self.__str__()

    def _get_match_weight(self, name, score):
        if (UNKNOWN_CN in name) and (score == 0):
            ww = self.UNKNOWN_SCORE
        else:
            # TODO: make weight: 1 - ln(score)
            ww = 1.0 - score

        return ww

    def _add_matches(self, column_names=None):
        """
        Add matches to the alignment graph.
        Schema matcher prediction is done here!
        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :return: Integration graph
        """
        logging.info("Prediction for the dataset using octopus...")
        print("Prediction for the dataset using octopus...")
        if self._predicted_df is not None:
            predicted_df = self._predicted_df
        else:
            predicted_df = self._octopus.matcher_predict(self._dataset)
        if column_names is None:
            column_names = [c.name for c in self._dataset.columns]

        logging.info("Adding match nodes to the alignment graph...")

        threshold = min(max(0.0, self._match_threshold), 1.0)  # threshold should be in range [0,1]
        new_graph = self.graph.copy()
        # lookup from data node labels to node ids
        data_node_map = defaultdict(list)
        for n_id, n_data in self.graph.nodes(data=True):
            if n_data["type"] == "DataNode":
                data_node_map[n_data["label"]].append(n_id)

        # we add matches only for those attributes which are mapped to existing data nodes
        available_types = set(data_node_map.keys())

        logging.info("Node map constructed: {}".format(data_node_map))

        if self._add_unknown and "scores_" + UNKNOWN_CN + "---" + UNKNOWN_DN not in predicted_df.columns:
            logging.info("Adding unknown scores!")
            predicted_df["scores_" + UNKNOWN_CN + "---" + UNKNOWN_DN] = 0.0

        cur_id = 1 + new_graph.number_of_nodes()  # we start nodes here from this number
        link_id = 1 + new_graph.number_of_edges()  # we start links here from this number
        scores_cols = [col for col in predicted_df.columns if col.startswith("scores_")]

        for idx, row in predicted_df.iterrows():
            if row["column_name"] not in column_names:
                continue
            node_data = {
                "type": "Attribute",
                "label": row["column_name"],
                "columnId": row["column_id"]
            }
            new_graph.add_node(cur_id, attr_dict=node_data)
            for score in scores_cols:
                # we ignore scores below threshold, but not unknowns!
                if row[score] <= threshold and (UNKNOWN_CN not in score):
                    logging.info("Ignoring <=threshold scores")
                    continue
                lab = score.split("scores_")[1]
                link_data = {
                    "type": "MatchLink",
                    "weight": self._get_match_weight(score, row[score]),  # inverse number
                    "label": row["column_name"]
                }
                if lab not in available_types:
                    logging.warning("Weird label {} for column".format(lab, row["column_name"]))
                    continue
                for target in data_node_map[lab]:
                    new_graph.add_edge(target, cur_id, key=link_id, attr_dict=link_data)
                    link_id += 1

            cur_id += 1

        print("Integration graph read: {} nodes, {} links".format(
            new_graph.number_of_nodes(), new_graph.number_of_edges()))
        logging.info("Integration graph read: {} nodes, {} links".format(
            new_graph.number_of_nodes(), new_graph.number_of_edges()))

        return new_graph

    def _alignment2dzn(self):
        """
        Convert the alignment graph into MiniZinc format
        :return: File name where the alignment graph is stored
        """
        alignment_path = os.path.join(
            self._path, "resources", "storage", "{}.alignment.graphml".format(self._octopus.id))
        # write alignment graph to the disk
        nx.write_graphml(self.graph, alignment_path)

        logging.info("Unnecessary conversion of the alignment graph:")
        with open(alignment_path) as f:
            content = f.read()
        g = Graph.from_graphml(content, self._simplify_graph)

        logging.info("Processing patterns")
        patterns = []
        patterns_w = []
        if self._patterns:
            with open(self._patterns) as f:
                content = f.readlines()
            for i in range(1, len(content)):
                l = content[i].strip().replace(' ', '').replace('"', '').replace('[', '').replace(']', '')
                l = l.split(',')
                # pattern,support,num_edges,edge_keys
                w = int(weight_conversion(float(l[1]))/self._pattern_sign)
                edgs = map(lambda x: g.edge_ids[x] if x in g.edge_ids else None, l[3:len(l)])
                edgs = [e for e in edgs if e is not None]
                if len(edgs) > 0:
                    patterns_w.append(w)
                    patterns.append(edgs)

        alignment_path = os.path.join(self._path, "resources", "storage", "{}.alignment.dzn".format(self._octopus.id))
        logging.info("Writing the alignment graph to dzn: {}".format(alignment_path))
        incr = lambda x: x + 1

        self._graph_node_map = dict([(n, int(g.node_names[n])) for n in g.node_types['ClassNode']] +\
                               [(n, int(g.node_names[n])) for n in g.node_types['DataNode']] +\
                               [(n, int(g.node_names[n])) for n in g.node_types['Attribute']])
        logging.info("      graph_node_map: {}".format(self._graph_node_map))

        with open(alignment_path, "w+") as f:
            f.write("{}\n".format(g.to_dzn()))
            # in python3 we need to get lists explicitly instead of iterators/generators
            cnodes = "cnodes = " + list2dznset(list(map(incr, [n for n in g.node_types['ClassNode']])))
            dnodes = "dnodes = " + list2dznset(list(map(incr, [n for n in g.node_types['DataNode']])))
            anodes = "anodes = " + list2dznset(list(map(incr, [n for n in g.node_types['Attribute']])))
            unknown = "nb_unknown_nodes = " + str(len(g.unk_info.data_nodes)) + ";"
            all_unk = "all_unk_nodes = " + list2dznlist(list(map(incr, g.unk_info.as_list())))

            nb_pats = "nb_patterns = " + str(len(patterns)) + ";"
            pats_w = "patterns_w = " + str(patterns_w) + ";"
            pats = "patterns = " + list2D2dznsetlist(patterns)

            f.write("{}\n".format(cnodes))
            f.write("{}\n".format(dnodes))
            f.write("{}\n".format(anodes))
            f.write("{}\n".format(unknown))
            f.write("{}\n".format(all_unk))
            f.write("{}\n".format(nb_pats))
            f.write("{}\n".format(pats_w))
            f.write("{}\n".format(pats))

        return alignment_path

    def _matches2dzn(self, column_names=None):
        """
        convert the "attribute" part of the graph (that is, all but the alignment) into a matching
        problem.
        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :return:
        """
        integration_graph = self._add_matches(column_names)
        self.graph = integration_graph
        integration_path = os.path.join(self._path, "resources", "storage", "{}.integration.graphml".format(
            self._dataset.id))
        nx.write_graphml(integration_graph, integration_path)

        logging.info("Converting matching problem to dzn...")
        with open(integration_path) as f:
            content = f.read()

        print("  construction...")
        g = Graph.from_graphml(content, self._simplify_graph)
        print("  graph constructed")
        logging.info("  graph constructed")

        attributes = g.node_types['Attribute']
        # update lookup dictionary
        for n in attributes:
            self._graph_node_map[int(g.node_names[n])] = int(g.node_names[n])

        max_val = 0
        min_att = min(attributes)
        doms = []
        for i, n in enumerate(attributes):
            # in python3 we need to explicitly get list
            d = list(map(lambda x: x + 1, [g.otherNode(e, n) for e in g.inc[n]]))
            max_val = max(max_val, max(d))
            doms.append(d)
        print("  doms constructed")
        logging.info("  doms constructed")

        matching_path = os.path.join(self._path, "resources", "storage", "{}.integration.dzn".format(
            self._dataset.id))
        with open(matching_path, "w+") as f:
            f.write("nbA = {};\n".format(len(attributes)))
            f.write("attribute_domains ={}\n".format(list2D2dznsetlist(list(doms))))
            # print("attribute_domains =", list2D2dznsetlist(list(doms)))
            logging.info("  attribute domains written")

            mc = [[0 for j in range(0, max_val + 1)] for i in attributes]
            logging.info("   max_val: {}, size mc: {}, min_att: {}".format(max_val, len(attributes), min_att))
            for i in range(len(g.edges)):
                e = g.edges[i]
                # logging.info("====> edge: {}".format(e))
                if e[0] in attributes:
                    mc[e[0] - min_att][1 + e[1]] = e[2]
                elif e[1] in attributes:
                    mc[e[1] - min_att][1 + e[0]] = e[2]
            logging.info("  mc constructed")

            mcs = "match_costs = [|"
            for i, r in enumerate(mc):
                c = str(r[1:]).replace("[", "").replace("]", "|") + ("];" if i == len(mc) - 1 else "\n")
                mcs += c
            logging.info("  mcs constructed")
            # print(mcs)
            f.write("{}\n".format(mcs))

            mcs = "match_costs_sorted = [|"
            for i, r in enumerate(mc):
                s = r[1:]
                x = sorted(range(len(s)), key=lambda k: s[k])
                x = list(map(lambda x: x + 1, x))
                c = str(x).replace("[", "").replace("]", "|") + ("];" if i == len(mc) - 1 else "\n")
                mcs += c
            logging.info("  mcs sorted constructed")
            f.write("{}\n".format(mcs))

            # print("attribute_names = ", list2dznlist([int(g.node_names[n]) for n in attributes]))
            f.write("attribute_names = {}\n".format(list2dznlist([int(g.node_names[n]) for n in attributes])))
            logging.info("  attribute names constructed")

        return matching_path

    def _generate_oznfzn(self, column_names=None):
        """

        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :return:
        """
        print("Generating .ozn and .fzn")

        base_path = os.path.join(self._path, "resources", "storage", "{}.integration".format(
            self._dataset.id))
        # alignment graph should be converted first before matches are added
        alignment_path = self._alignment2dzn()
        match_path = self._matches2dzn(column_names)
        logging.info("Obtaining fzn and dzn files...".format(
            self._dataset.id, self._octopus.id))

        # args = ["" for _ in range(7)]
        # args[0] = "{}".format(self.mzn2fzn_exec)
        # args[1] = "-I {}".format(os.path.join(self.chuffed_path, "globals"))  # include path
        # args[2] = "{}".format(self.model_mzn)  # model file
        # args[3] = "{}".format(alignment_path)  # data files
        # args[4] = "{}".format(match_path)  # data files
        # args[5] = "-o {}.fzn".format(base_path)  # output files
        # args[6] = "-O {}.ozn".format(base_path)  # output files

        command = "{} -I {}/globals/ {} {} {} -o {}.fzn " \
                  "-O {}.ozn".format(self._mzn2fzn_exec, self._chuffed_path, self._model_mzn,
                                     alignment_path, match_path, base_path, base_path)

        print("Executing command: {}".format(command))
        # output = subprocess.check_call(args)
        output = subprocess.call(command, shell=True)

        print("Done calling")
        print(output)

        return base_path

    def _find_link(self, start, end, weight):
        """
        Helper method to find a link (start, end) in the integration graph with weight.
        Check both directions for the link.
        :param start:
        :param end:
        :param weight:
        :return:
        """
        # making sure that nothing is done to the graph
        copy_graph = self.graph.copy()

        def find_weight(links):
            logging.debug(" finding weight {} among links {}".format(weight, links))
            if len(links) == 1:
                return links.popitem()[1]
            elif len(links) > 1:
                # we need to find the link with weight
                # we take the first match!
                for key, val in links.items():
                    if abs(val["weight"] - weight) < 0.001:
                        return val
            else:
                logging.error("No info for the link")
                raise Exception("No info for the link")

        if start not in copy_graph and end not in copy_graph:
            logging.error("Start and end nodes not in the integration graph.")
            print("Start and end nodes not in the integration graph.")
            return None
        if end not in copy_graph[start]:
            if start not in copy_graph[end]:
                logging.error("Link cannot be found in the integration graph.")
                print("Link cannot be found in the integration graph.")
                return None
            else:
                link = find_weight(copy_graph[end][start])
                link["source"] = end
                link["target"] = start

        else:
            link = find_weight(copy_graph[start][end])
            link["source"] = start
            link["target"] = end

        link["status"] = "Predicted"
        return link

    @staticmethod
    def _get_attr(ag, attr_name):
        """
        Get value of the graph attribute.
        If not available, None is returned
        :param ag: Graphviz object
        :param attr_name: string
        :return:
        """
        if attr_name in ag.graph_attr:
            return ag.graph_attr[attr_name]
        return None

    def _digraph_to_ssd(self, digraph, dot_file=None, optimal_flag=False):
        """
        Convert digraph string to ssd request
        :param digraph: string of a digraph
        :param dot_file: optional file name to write the solution to the system to a dot file
        :return:
        """
        # for some unknown reason stuff disappears from the graph
        copy_graph = self.graph.copy()
        logging.debug("Converting digraph {} to ssd".format(digraph))
        ag = AGraph(digraph)
        if dot_file:
            ag.write(dot_file)  # write the dot file to the system
        # convert to SsdRequest: name, ontologies, semanticModel, mappings
        start_time = time.time()
        t = float(self._get_attr(ag, "time"))
        cost = self._get_attr(ag, "cost")
        match_score = self._get_attr(ag, "match_score")
        obj = self._get_attr(ag, "obj")
        ssd_request = {
            "ontologies": [onto.id for onto in self._octopus.ontologies],
            "name": self._dataset.filename,
            "mappings": [],
            "semanticModel": {"nodes": [], "links": []},
            "time": t,
            "cost": cost,
            "match_score": match_score,
            "objective": obj,
            "optimalFlag": optimal_flag
        }

        # mappings: [{"attribute": column_id, "node": data_node_id}]
        # semanticModel: {
        #   "nodes": [{"id": ,"label": "", "prefix": "", "status": "","type": ""}],
        #   "links": [{ "id": , "label": "", "prefix": "", "source": , "status": "", "target": , "type": ""}]
        # }

        attributes = [self._graph_node_map[int(n)] for n in ag.iternodes() if n.attr["shape"] == "hexagon"]
        logging.debug("Attributes obtained: {}".format(attributes))
        sm_nodes = [self._graph_node_map[int(n)] for n in ag.iternodes() if n.attr["shape"] != "hexagon"]
        logging.debug("Nodes for the semantic model obtained: {}".format(sm_nodes))

        for ind, ed in enumerate(ag.edges_iter()):
            logging.debug("   working on edge {}: {}".format(ind, ed))
            start = self._graph_node_map[int(ed[0])]
            end = self._graph_node_map[int(ed[1])]
            if start not in attributes and end not in attributes:
                # add a link to the semantic model
                weight = float(ed.attr["label"].replace("w=", ""))
                logging.debug("     sm link: start={}, end={}, weight={}".format(start, end, weight))
                link = self._find_link(start, end, weight)
                logging.debug("     sm link found")
                link["id"] = ind
                ssd_request["semanticModel"]["links"].append(link)
                logging.debug("     adding link: {}".format(link))
            else:
                # add a mapping
                logging.debug("     adding mapping: start={}, end={}".format(start, end))
                if start in attributes:
                    column_id = copy_graph.node[start]["columnId"]
                    ssd_request["mappings"].append({"attribute": column_id, "node": end})
                    logging.debug("     adding mapping: column={}, end={}".format(column_id, end))
                elif end in attributes:
                    column_id = copy_graph.node[end]["columnId"]
                    ssd_request["mappings"].append({"attribute": column_id, "node": start})
                    logging.debug("     adding mapping: column={}, start={}".format(column_id, start))
                else:
                    logging.error("Something wrong with the edge: {}".format(ed))

        logging.debug("Links have been added")

        for n in sm_nodes:
            # add nodes to the semantic model
            node = copy_graph.node[n]
            node["status"] = "Predicted"
            node["id"] = n
            ssd_request["semanticModel"]["nodes"].append(node)

        logging.debug("Conversion finished")
        conv_time = time.time() - start_time
        ssd_request["extended_time"] = ssd_request["time"] + conv_time

        # getting the graph to its original state
        # self.graph = copy_graph

        return ssd_request

    def _process_solution(self, output, dot_file=None):
        """
        Here we want to convert to some form of SSD.
        In particular we convert to Ssd Request and then dump to json.
        To obtain SSD object: SSD().update(json.loads(ssd_request), sn._datasets, sn._ontologies)
        :param output: byte stuff returned by calling Chuffed solver via subprocess
        :param dot_file: optional file name to write the solution to the system to a dot file
        :return: List of Json for SsdRequest
        """
        print("Converting Chuffed solution to ssd request")
        logging.info("Converting Chuffed solution to ssd request")
        # process output from byte to string
        output = output.decode("ascii")
        pat = re.compile("digraph \{.+?\}", re.DOTALL)
        logging.debug("Chuffed output: {}".format(output))
        optimal_flag = not("Time limit exceeded" in output)

        try:
            # get graphviz digraph
            # we take the last one since it's supposed to be the most optimal found so far
            # logging.debug("Chuffed output: ", output)
            # digraph = pat.findall(output)[-1]
            # ssd_request = self._digraph_to_ssd(digraph, dot_file)
            # # to return json:
            # return json.dumps(ssd_request)
            result = [json.dumps(self._digraph_to_ssd(digraph, dot_file, optimal_flag)) for digraph in pat.findall(output)]

            return result

        except Exception as e:
            print("Conversion of chuffed solution failed: {}".format(str(e)))
            logging.error("Conversion of chuffed solution failed: {}".format(str(e)))
            return

    def solve(self, column_names=None, dot_file=None,
              soft_assumptions=False,
              relax_every=2):
        """
        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :param dot_file: optional file name to write the dot file to the system
        :param soft_assumptions: boolean to drop the requirement that all attribute must match a data node
        :param relax_every: integer seconds used when soft_assmuptions is True; it will ensure that the requirement
        for attributes to match a data node will be dropped every interval of seconds for half of attributes
        :return: List of Json for SsdRequest
        """
        print("Solving the integration problem with Chuffed...")
        logging.info("Solving the integration problem with Chuffed...")
        base_path = self._generate_oznfzn(column_names)

        logging.info("Solving the problem for dataset {} using octopus {}".format(
            self._dataset.id, self._octopus.id))
        if soft_assumptions:
            command = "{}/fzn_chuffed -verbosity=2 -time_out={} -lazy=true " \
                      "-soft_assumptions=True -relax_every={} " \
                      "-steinerlp=true {}.fzn | {} {}.ozn".format(self._chuffed_path,
                                                                  self._timeout,
                                                                  relax_every,
                                                                  base_path,
                                                                  self._solns2out_exec,
                                                                  base_path)
        else:
            command = "{}/fzn_chuffed -verbosity=2 -time_out={} -lazy=true " \
                      "-steinerlp=true {}.fzn | {} {}.ozn".format(self._chuffed_path,
                                                                  self._timeout,
                                                                  base_path,
                                                                  self._solns2out_exec,
                                                                  base_path)
        print("Executing command {}".format(command))
        logging.info("Executing command {}".format(command))
        try:
            output = subprocess.check_output(command, shell=True)
        except Exception as esc:
            print(esc)
            logging.error("Chuffed failed: {}".format(esc))
            return

        return self._process_solution(output, dot_file)
