"""
License...
"""

import logging
import networkx as nx
import yaml
import os
from collections import defaultdict
from .minizinc import *
import subprocess
import re
import json
from pygraphviz import AGraph
from serene import SSD

class IntegrationGraph(object):
    """
    Integration graph to map a relaitonal data source onto an ontology using a CP solver
    """
    def __init__(self, octopus, dataset, match_threshold=0.0, simplify_graph=False):
        """
        Octopus should be trained.
        :param octopus: Octopus object which is used to integrate the new data source
        :param dataset: Dataset object for which integration is performed
        :param match_threshold: threshold on match links
        :param simplify_graph: Remove edges that are heavier than some path between its endpoints
        :return:
        """
        logging.info("Initializing integration graph for the dataset {} using octopus {}".format(
            dataset.id, octopus.id))
        self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stp")
        self.octopus = octopus
        self.dataset = dataset
        self.match_threshold = match_threshold  # discard matches with confidence score < threshold
        self.simplify_graph = simplify_graph

        self.graph = octopus.get_alignment()  # the integration graph contains the alignment graph at the beginning
        self.chuffed_path, self.mzn2fzn_exec, self.model_mzn, self.solns2out_exec, self.timeout = \
            self._read_config()

    def _read_config(self):
        logging.info("Reading configuration parameters for the integration graph")

        with open(os.path.join(self.path, 'config.yaml'), 'r') as stream:
            config = yaml.load(stream)

        return config["chuffed-path"], config["mzn2fzn-exec"], \
               config["model-path"], config["solns2out-exec"], config["timeout"]

    def __str__(self):
        return "<IntegrationGraph({}, {}, match_threshold={}, simplify={})>".format(self.octopus, self.dataset,
                                                                                    self.match_threshold, self.simplify_graph)

    def __repr__(self):
        return self.__str__()

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
        predicted_df = self.octopus.matcher_predict(self.dataset)
        # TODO: handle unknowns somehow...
        if column_names is None:
            column_names = [c.name for c in self.dataset.columns]

        logging.info("Adding match nodes to the alignment graph...")

        threshold = min(max(0.0, self.match_threshold), 1.0)  # threshold should be in range [0,1]
        new_graph = self.graph.copy()
        # lookup from data node labels to node ids
        data_node_map = defaultdict(list)
        for n_id, n_data in self.graph.nodes(data=True):
            if n_data["type"] == "DataNode":
                data_node_map[n_data["label"]].append(n_id)

        # we add matches only for those attributes which are mapped to existing data nodes
        available_types = set(data_node_map.keys())

        logging.info("Node map constructed: {}".format(data_node_map))

        cur_id = 1 + new_graph.number_of_nodes()  # we start nodes here from this number
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

    def _alignment2dzn(self):
        """
        Convert the alignment graph into MiniZinc format
        :return: File name where the alignment graph is stored
        """
        alignment_path = os.path.join(self.path, "resources", "storage", "{}.alignment.graphml".format(self.octopus.id))
        # write alignment graph to the disk
        nx.write_graphml(self.graph, alignment_path)

        logging.info("Unnecessary conversion of the alignment graph:")
        content = ""
        with open(alignment_path) as f:
            content = f.read()
        g = Graph.from_graphml(content, self.simplify_graph)

        alignment_path = os.path.join(self.path, "resources", "storage", "{}.alignment.dzn".format(self.octopus.id))
        logging.info("Writing the alignment graph to dzn: {}".format(alignment_path))
        incr = lambda x: x + 1
        with open(alignment_path, "w+") as f:
            f.write("{}\n".format(g.to_dzn()))
            # in python3 we need to get lists explicitly instead of iterators/generators
            cnodes = "cnodes = " + list2dznset(list(map(incr, [int(g.node_names[n]) for n in g.node_types['ClassNode']])))
            dnodes = "dnodes = " + list2dznset(list(map(incr, [int(g.node_names[n]) for n in g.node_types['DataNode']])))
            anodes = "anodes = " + list2dznset(list(map(incr, [int(g.node_names[n]) for n in g.node_types['Attribute']])))
            f.write("{}\n".format(cnodes))
            f.write("{}\n".format(dnodes))
            f.write("{}\n".format(anodes))

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
        integration_path = os.path.join(self.path, "resources", "storage", "{}.integration.graphml".format(
            self.dataset.id))
        nx.write_graphml(integration_graph, integration_path)

        logging.info("Converting matching problem to dzn...")
        content = ""
        with open(integration_path) as f:
            content = f.read()

        g = Graph.from_graphml(content, self.simplify_graph)

        attributes = g.node_types['Attribute']

        max_val = 0
        min_att = min(attributes)
        doms = []
        for i, n in enumerate(attributes):
            # in python3 we need to explicitly get list
            d = list(map(lambda x: x + 1, [g.otherNode(e, n) for e in g.inc[n]]))
            max_val = max(max_val, max(d))
            doms.append(d)

        matching_path = os.path.join(self.path, "resources", "storage", "{}.integration.dzn".format(
            self.dataset.id))
        with open(matching_path, "w+") as f:
            f.write("nbA = {};\n".format(len(attributes)))
            f.write("attribute_domains ={}\n".format(list2D2dznsetlist(list(doms))))
            # print("attribute_domains =", list2D2dznsetlist(list(doms)))

            mc = [[0 for j in range(0, max_val + 1)] for i in attributes]
            for i in range(0, len(g.edges)):
                e = g.edges[i]
                if e[0] in attributes:
                    mc[e[0] - min_att][1 + e[1]] = e[2]
                elif e[1] in attributes:
                    mc[e[1] - min_att][1 + e[0]] = e[2]

            mcs = "match_costs = [|"
            for i, r in enumerate(mc):
                c = str(r[1:]).replace("[", "").replace("]", "|") + ("];" if i == len(mc) - 1 else "\n")
                mcs += c
            # print(mcs)
            f.write("{}\n".format(mcs))

            mcs = "match_costs_sorted = [|"
            for i, r in enumerate(mc):
                s = r[1:]
                x = sorted(range(len(s)), key=lambda k: s[k])
                x = list(map(lambda x: x + 1, x))
                c = str(x).replace("[", "").replace("]", "|") + ("];" if i == len(mc) - 1 else "\n")
                mcs += c
            # print(mcs)
            f.write("{}\n".format(mcs))

            # print("attribute_names = ", list2dznlist([int(g.node_names[n]) for n in attributes]))
            f.write("attribute_names = {}\n".format(list2dznlist([int(g.node_names[n]) for n in attributes])))

        return matching_path

    def _generate_oznfzn(self, column_names=None):
        """

        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :return:
        """
        print("Generating .ozn and .fzn")

        base_path = os.path.join(self.path, "resources", "storage", "{}.integration".format(
            self.dataset.id))
        # alignment graph should be converted first before matches are added
        alignment_path = self._alignment2dzn()
        match_path = self._matches2dzn(column_names)
        logging.info("Obtaining fzn and dzn files...".format(
            self.dataset.id, self.octopus.id))

        # args = ["" for _ in range(7)]
        # args[0] = "{}".format(self.mzn2fzn_exec)
        # args[1] = "-I {}".format(os.path.join(self.chuffed_path, "globals"))  # include path
        # args[2] = "{}".format(self.model_mzn)  # model file
        # args[3] = "{}".format(alignment_path)  # data files
        # args[4] = "{}".format(match_path)  # data files
        # args[5] = "-o {}.fzn".format(base_path)  # output files
        # args[6] = "-O {}.ozn".format(base_path)  # output files

        command = "{} -I {}/globals/ {} {} {} -o {}.fzn " \
                  "-O {}.ozn".format(self.mzn2fzn_exec, self.chuffed_path, self.model_mzn,
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
        def find_weight(links):
            if len(links) == 1:
                return links[0]
            else:
                # we need to find the link with weight
                # we take the first match!
                for key, l in links.items():
                    if abs(l["weight"] - weight) < 0.001:
                        return l

        if start not in self.graph and end not in self.graph:
            logging.error("Start and end nodes not in the integration graph.")
            print("Start and end nodes not in the integration graph.")
            return None
        if end not in self.graph[start]:
            if start not in self.graph[end]:
                logging.error("Link cannot be found in the integration graph.")
                print("Link cannot be found in the integration graph.")
                return None
            else:
                link = find_weight(self.graph[end][start])
                link["source"] = end
                link["target"] = start

        else:
            link = find_weight(self.graph[start][end])
            link["source"] = start
            link["target"] = end

        link["status"] = "Predicted"
        return link

    def _digraph_to_ssd(self, digraph, dot_file=None):
        """
        Convert digraph string to ssd request
        :param digraph: string of a digraph
        :param dot_file: optional file name to write the solution to the system to a dot file
        :return:
        """
        ag = AGraph(digraph)
        if dot_file:
            ag.write(dot_file)  # write the dot file to the system
        # convert to SsdRequest: name, ontologies, semanticModel, mappings
        ssd_request = {
            "ontologies": [onto.id for onto in self.octopus.ontologies],
            "name": self.dataset.filename,
            "mappings": [],
            "semanticModel": {"nodes": [], "links": []}
        }

        # mappings: [{"attribute": column_id, "node": data_node_id}]
        # semanticModel: {
        #   "nodes": [{"id": ,"label": "", "prefix": "", "status": "","type": ""}],
        #   "links": [{ "id": , "label": "", "prefix": "", "source": , "status": "", "target": , "type": ""}]
        # }

        attributes = [int(n) for n in ag.iternodes() if n.attr["shape"] == "hexagon"]
        sm_nodes = [int(n) for n in ag.iternodes() if n.attr["shape"] != "hexagon"]
        for ind, e in enumerate(ag.edges_iter()):
            start = int(e[0])
            end = int(e[1])
            if start not in attributes and end not in attributes:
                # add a link to the semantic model
                weight = float(e.attr["label"].replace("w=", ""))
                link = self._find_link(start, end, weight)
                link["id"] = ind
                ssd_request["semanticModel"]["links"].append(link)
            else:
                # add a mapping
                if start in attributes:
                    column_id = self.graph.node[start]["columnId"]
                    ssd_request["mappings"].append({"attribute": column_id, "node": end})
                elif end in attributes:
                    column_id = self.graph.node[end]["columnId"]
                    ssd_request["mappings"].append({"attribute": column_id, "node": start})

        for n in sm_nodes:
            # add nodes to the semantic model
            node = self.graph.node[n]
            node["status"] = "Predicted"
            node["id"] = n
            ssd_request["semanticModel"]["nodes"].append(node)

        return ssd_request

    def _process_solution(self, output, dot_file=None):
        """
        Here we want to convert to some form of SSD.
        In particular we convert to Ssd Request and then dump to json.
        To obtain SSD object: SSD().update(json.loads(ssd_request), sn._datasets, sn._ontologies)
        :param output: byte stuff returned by calling Chuffed solver via subprocess
        :param dot_file: optional file name to write the solution to the system to a dot file
        :return: Json for SsdRequest
        """
        print("Converting Chuffed solution to ssd request")
        logging.info("Converting Chuffed solution to ssd request")
        # process output from byte to string
        output = output.decode("ascii")
        pat = re.compile("digraph \{.+?\}", re.DOTALL)

        try:
            # get graphviz digraph
            # we take the last one since it's supposed to be the most optimal found so far
            digraph = pat.findall(output)[-1]
            ssd_request = self._digraph_to_ssd(digraph)
            # to return json:
            return json.dumps(ssd_request)

        except Exception as e:
            print("Conversion of chuffed solution failed {}".format(e.args[0]))
            logging.error("Conversion of chuffed solution failed {}".format(e.args[0]))
            return

    def solve(self, column_names=None, dot_file=None):
        """
        :param column_names: list of column names which should be present in the solution;
                if None, all columns will be taken
        :param dot_file: optional file name to write the dot file to the system
        :return:
        """
        print("Solving the integration problem with Chuffed...")
        logging.info("Solving the integration problem with Chuffed...")
        base_path = self._generate_oznfzn(column_names)

        logging.info("Solving the problem for dataset {} using octopus {}".format(
            self.dataset.id, self.octopus.id))
        command = "{}/fzn_chuffed -verbosity=2 -time_out={} -lazy=true " \
                  "-steinerlp=true {}.fzn | {} {}.ozn".format(self.chuffed_path,
                                                              self.timeout,
                                                              base_path,
                                                              self.solns2out_exec,
                                                              base_path)
        print("Executing command {}".format(command))
        logging.info("Executing command {}".format(command))
        try:
            output = subprocess.check_output(command, shell=True)
        except Exception as e:
            print(e)
            logging.error("Chuffed failed: {}".format(e))
            return


        return self._process_solution(output, dot_file)
