# ===========
#    SETUP
# ===========
try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene

import os
import time
import tarfile
import tempfile
from pprint import pprint
import networkx as nx
import pandas as pd
import json

from serene import SSD, Status, DataProperty, Mapping, ObjectProperty, Column, Class, DataNode, ClassNode
from serene.elements import SubClassLink, ObjectLink, ColumnLink
from serene.elements.semantics.base import KARMA_DEFAULT_NS, DEFAULT_NS, ALL_CN, OBJ_PROP, UNKNOWN_CN, UNKNOWN_DN

from stp.alignment import read_karma_graph, add_matches, convert_ssd, to_graphviz, convert_karma_graph
from collections import defaultdict

import logging
from logging.handlers import RotatingFileHandler
project_path = "/home/natalia/PycharmProjects/serene-python-client"
# setting up the logging
log_file = os.path.join(project_path, "stp", "resources", 'benchmark_stp.log')
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

# =======================
#
#  Step 1: Start with a connection to the server...
#
# =======================
sn = serene.Serene(
    host='127.0.0.1',
    port=8080,
)
print(sn)

benchmark_path = os.path.join(project_path, "tests", "resources", "museum_benchmark")
print("benchmark_path: ", benchmark_path)

print("========Optional cleaning=============")
# Removes all server elements
for octo in sn.octopii.items:
    sn.octopii.remove(octo)

for ssd in sn.ssds.items:
    sn.ssds.remove(ssd)

for ds in sn.datasets.items:
    sn.datasets.remove(ds)

for on in sn.ontologies.items:
    sn.ontologies.remove(on)

# =======================
#
#  Step 2: Add ontologies from the museum benchmark
#
# =======================
owl_dir = os.path.join(benchmark_path, "owl")

ontologies = []
for path in os.listdir(owl_dir):
    f = os.path.join(owl_dir, path)
    ontologies.append(sn.ontologies.upload(f))

# add Unknown class
# unknown_owl = os.path.join("unknown", "unknown_class.ttl")
unknown_owl = os.path.join("stp","unknown", "unknown_class.ttl")
ontologies.append(sn.ontologies.upload(unknown_owl))

print("*********** Ontologies from museum benchmark")
for onto in ontologies:
    print(onto)
print()

# =======================
#
#  Step 3: Add datasets from the museum benchmark and corresponding semantic source descriptions
#
# =======================
dataset_dir = os.path.join(benchmark_path, "dataset")
ssd_dir = os.path.join(benchmark_path, "ssd")

datasets = [] # list of museum datasets
ssds = [] # list of semantic source descriptions from museum benchmark

# we need to unzip first the data
if os.path.exists(os.path.join(dataset_dir, "data.tar.gz")):
    with tarfile.open(os.path.join(dataset_dir, "data.tar.gz")) as f:
        f.extractall(path=dataset_dir)

# input("Press enter to upload datasets and ssds...")

for ds in os.listdir(dataset_dir):
    if ds.endswith(".gz"):
        continue
    ds_f = os.path.join(dataset_dir, ds)
    ds_name = os.path.splitext(ds)[0]
    # associated semantic source description for this dataset
    ssd_f = os.path.join(ssd_dir, ds_name + ".ssd")

    print("Adding dataset: " + str(ds_f))
    dataset = sn.datasets.upload(ds_f, description="museum_benchmark")
    datasets.append(dataset)

    print("Adding ssd: " + str(ssd_f))
    new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
    # _logger.info("+++++++++++ obtained ssd json")
    # _logger.info(new_json)
    if len(new_json["mappings"]) < 1:
        print("     --> skipping this file...")
        continue
    else:
        empty_ssd = SSD(dataset, ontologies)
        ssd = empty_ssd.update(new_json, sn.datasets, sn.ontologies)
        ssds.append(sn.ssds.upload(ssd))
    # we remove the csv dataset
    os.remove(ds_f)


# =======================
#
#  Step 5: Add unknown mappings to ssds
#
# ======================
from copy import deepcopy

new_ssds = [deepcopy(s) for s in sn.ssds.items]

for s in new_ssds:
    s = s.fill_unknown()
    # s.show()
    # input("Press enter to continue...")

for s in new_ssds:
    f = [d for d in s.data_nodes if d.full_label == UNKNOWN_CN + "." + UNKNOWN_DN]
    if len(f)<2:
        print(s)
        print(f)
        print()


# upload ssds with unknown mappings
new_ssds = [sn.ssds.upload(s) for s in new_ssds]

for s in new_ssds:
    f = [d for d in s.data_nodes if d.full_label == UNKNOWN_CN + "." + UNKNOWN_DN]
    if len(f)<2:
        print(s)
        print(f)
        print()


# =======================
#
#  Leave-one-out experiments
#
# =======================

def create_octopus(ssd_range, sample):
    octo_local = sn.Octopus(
        ssds=[ssd_range[i] for i in sample],
        ontologies=ontologies,
        name='octopus-unknown',
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
            "sizeWeight": 0.5,
            "numSemanticTypes": 10,
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
                # ,"mean-character-cosine-similarity-from-class-examples"
                # ,"min-editdistance-from-class-examples"
                # ,"min-wordnet-jcn-distance-from-class-examples"
                # ,"min-wordnet-lin-distance-from-class-examples"
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
    octo = sn.octopii.upload(octo_local)

    start = time.time()
    print()
    print("Next we can train the Octopus")
    print("The initial state for {} is {}".format(octo.id, octo.state))
    print("Training...")
    octo.train()
    print("Done in: {}".format(time.time() - start))
    print("The final state for {} is {}".format(octo.id, octo.state))

    return octo


import stp.integration as integ
from stp.alignment import convert_ssd, to_graphviz, process_unknown
from importlib import reload
import csv
import stp.alignment as al

reload(integ)
reload(al)


def do_chuffed(ch_octopus, ch_dataset, ch_column_names, ch_ssd, orig_ssd,
               train_flag=False, chuffed_path=None, simplify=True, patterns=None):
    res = []
    try:
        integrat = integ.IntegrationGraph(octopus=ch_octopus, dataset=ch_dataset,
                                          match_threshold=0.01, simplify_graph=simplify,
                                          patterns=patterns)
        start = time.time()
        if chuffed_path:
            integrat._chuffed_path = chuffed_path  # change chuffed solver version
        solution = integrat.solve(ch_column_names)
        runtime = time.time() - start
        # print(solution)
        if len(solution) < 1:
            logging.error("Chuffed found no solutions for ssd {}".format(ssd.id))
            raise Exception("Chuffed found no solutions!")
        for idx, sol in enumerate(solution):
            # convert solution to SSD
            cp_ssd = SSD().update(json.loads(sol), sn._datasets, sn._ontologies)
            eval = cp_ssd.evaluate(ch_ssd, False, True)

            try:
                comparison = sn.ssds.compare(cp_ssd, ch_ssd, False, False)
            except:
                comparison = {"jaccard": -1, "precision": -1, "recall": -1}

            res += [[ch_octopus.id, ch_dataset.id, orig_ssd.name, orig_ssd.id, "chuffed", idx, "success",
                   eval["jaccard"], eval["precision"], eval["recall"],
                   comparison["jaccard"], comparison["precision"], comparison["recall"],
                   train_flag, runtime, chuffed_path]]
    except Exception as e:
        print("CP solver failed to find solution: {}".format(e))
        res += [[ch_octopus.id, ch_dataset.id, orig_ssd.name, orig_ssd.id, "chuffed", 0, "fail", 0, 0, 0, 0, 0, 0,
               train_flag, time.time() - start, chuffed_path]]
    return res


def do_karma(k_octopus, k_dataset, k_ssd, orig_ssd,
             train_flag=False):
    try:
        start = time.time()
        karma_ssd = k_octopus.predict(k_dataset)[0].ssd
        runtime = time.time() - start
        eval = karma_ssd.evaluate(k_ssd, False, True)

        try:
            comparison = sn.ssds.compare(karma_ssd, k_ssd, False, False)
        except:
            comparison = {"jaccard": -1, "precision": -1, "recall": -1}

        return [[k_octopus.id, k_dataset.id, orig_ssd.name, orig_ssd.id, "karma", 0, "success",
               eval["jaccard"], eval["precision"], eval["recall"],
               comparison["jaccard"], comparison["precision"], comparison["recall"],
               train_flag, runtime, ""]]
    except Exception as e:
        print("Karma failed to find solution: {}".format(e))
        return [[k_octopus.id, k_dataset.id, orig_ssd.name, orig_ssd.id, "karma", 0, "fail", 0, 0, 0, 0, 0, 0,
                train_flag, 0, ""]]


folder = os.path.join(project_path, "stp", "resources", "unknown_run")
chuffed_paths = ["/home/natalia/PycharmProjects/serene-python-client/stp/minizinc/chuffed-rel2onto_20170718",
                 "/home/natalia/PycharmProjects/serene-python-client/stp/minizinc/chuffed-rel2onto_20170722",
                 "/home/natalia/PycharmProjects/serene-python-client/stp/minizinc/chuffed-rel2onto_20170728"]

result_csv = os.path.join(project_path, "stp", "resources", "unknown_benchmark_results_new.csv")
with open(result_csv, "w+") as f:
    csvf = csv.writer(f)
    csvf.writerow(["octopus", "dataset", "name", "ssd", "method", "solution", "status",
                              "jaccard", "precision", "recall",
                              "karma_jaccard", "karma_precision", "karma_recall",
                              "train_flag", "time", "chuffed_path"])  # header

    sample_range = list(range(len(new_ssds)))
    benchmark_results = [["octopus", "dataset", "name", "ssd", "method", "solution", "status",
                          "jaccard", "precision", "recall",
                          "karma_jaccard", "karma_precision", "karma_recall",
                          "train_flag", "time", "chuffed_path"]]

    print("Starting benchmark")

    for cur_id, _ in enumerate(new_ssds):
        print()
        print("--> {} Leave one out".format(cur_id))
        # specify train sample
        train_sample = sample_range[:cur_id] + sample_range[cur_id+1:]
        octo = create_octopus(new_ssds, train_sample)
        octo_csv = os.path.join(project_path, "stp", "resources", "storage",
                                "patterns.{}.csv".format(octo.id))
        octo_patterns = octo.get_patterns(octo_csv)
        print("Uploaded octopus:", octo)

        print("Getting labels in the training set")
        train_labels = []
        for i in train_sample:
            train_labels += [d.full_label for d in new_ssds[i].data_nodes]
        train_labels = set(train_labels)

        for i, ssd in enumerate(new_ssds):
            dataset = [ds for ds in datasets if ds.filename == ssd.name+".csv"][0]
            # we tell here the correct columns
            column_names = [c.name for c in ssd.columns]
            print("Chosen dataset: ", dataset)
            print("Chosen SSD id: ", ssd.id)

            # we should do prediction outside of solvers so that solvers use the cached results!
            print("Doing schema matcher part")
            df = octo.matcher_predict(dataset)

            cor_ssd = al.process_unknown(ssd, train_labels)

            # cp solver approach
            for chuffed in chuffed_paths:
                print("---> trying chuffed {}".format(chuffed))
                res = do_chuffed(octo, dataset, column_names, cor_ssd, ssd,
                                 train_flag=(i in train_sample),
                                 chuffed_path=chuffed, simplify=True, patterns=octo_patterns)

                benchmark_results += res
                csvf.writerows(res)

            # karma approach
            print("---> trying karma")
            res = do_karma(octo, dataset, cor_ssd, ssd, train_flag=(i in train_sample))
            benchmark_results += res
            csvf.writerows(res)
            f.flush()

print("Benchmark finished!")
print(benchmark_results)

# =======================
#
#  Step 6: Specify training and test samples
#
# =======================
train_sample = []
test_sample = []

# s16 will be test
for i, ssd in enumerate(new_ssds):
    # if "s07" in ssd.dataset.filename:
    #     test_sample.append(i)
    # else:
    #     # if "s07" in ds.filename or "s08" in ds.filename:
    #     train_sample.append(i)
    if i < 14:
        test_sample.append(i)
    else:
        train_sample.append(i)

# train_sample = train_sample[:5]

print("Indexes for training sample: ", train_sample)
print("Indexes for testing sample: ", test_sample)

print("Original ssds in train")
train_orig_ssd = []
for i in train_sample:
    s = [ssd for ssd in ssds if ssd.name == new_ssds[i].name]
    print("ssd: ", s[0].id, ", name: ", s[0].name, len(s))
    train_orig_ssd.append(s[0].id)

print(sorted(train_orig_ssd))

# =======================
#
#  Step 7: Create octopus
#
# =======================

octo_local = sn.Octopus(
    ssds=[new_ssds[i] for i in train_sample],
    ontologies=ontologies,
    name='octopus-unknown',
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
        "sizeWeight": 0.5,
        "numSemanticTypes": 10,
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
            ,"prop-instances-per-class-in-knearestneighbours"
            # ,"mean-character-cosine-similarity-from-class-examples"
            # ,"min-editdistance-from-class-examples"
            # ,"min-wordnet-jcn-distance-from-class-examples"
            # ,"min-wordnet-lin-distance-from-class-examples"
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
octo = sn.octopii.upload(octo_local)

print("Uploaded octopus:", octo)

# =======================
#
# Step 6. Train
#
# =======================

start = time.time()
print()
print("Next we can train the Octopus")
print("The initial state for {} is {}".format(octo.id, octo.state))
print("Training...")
octo.train()
print("Done in: {}".format(time.time() - start))
print("The final state for {} is {}".format(octo.id, octo.state))

input("Press enter to continue...")

# =======================
#
# Step 7. Generate benchmark for CP solver
#
# =======================

import stp.integration as integ
from stp.alignment import convert_ssd, to_graphviz, process_unknown
from importlib import reload

import stp.alignment as al

reload(integ)
reload(al)

print("Starting generation")

print("Getting labels in the training set")
train_labels = []
for i in train_sample:
    train_labels += [d.full_label for d in new_ssds[i].data_nodes]
train_labels = set(train_labels)


folder = os.path.join(project_path, "stp", "resources", "unknown_thresh01")
for dataset in datasets:
    ssd = [s for s in new_ssds if s.name+".csv" == dataset.filename][0]
    # we tell here the correct columns
    column_names = [c.name for c in ssd.columns]
    print("Chosen dataset: ", dataset)
    print("Chosen SSD id: ", ssd.id)

    # we should do prediction outside of solvers so that solvers use the cached results!
    print("Doing schema matcher part")
    df = octo.matcher_predict(dataset)

    # cp solver approach
    print("---> Converting")
    integrat = integ.IntegrationGraph(octopus=octo, dataset=dataset,
                                      match_threshold=0.01, simplify_graph=True)

    # write alignment graph
    align_path = os.path.join(folder, "{}.alignment".format(ssd.id))
    nx.write_graphml(integrat.graph, align_path+".graphml")
    to_graphviz(integrat.graph, out_file=align_path+".dot", prog="dot")
    print("---> alignment written")

    # write integration graph
    integration_graph = integrat._add_matches(column_names)
    integration_path = os.path.join(folder, "{}.integration".format(ssd.id))
    nx.write_graphml(integration_graph, integration_path + ".graphml")
    to_graphviz(integration_graph, out_file=integration_path + ".dot", prog="dot")
    print("---> integration written")

    # write ground truth
    ssd_path = os.path.join(folder, "{}.ssd".format(ssd.id))
    unknown_ssd = al.process_unknown(ssd, train_labels)
    ssd_nx = al.convert_ssd(unknown_ssd, integration_graph, ssd_path + ".graphml")
    to_graphviz(ssd_nx, out_file=ssd_path + ".dot", prog="dot")
    print("---> ssd written")


print("Finished generation")

print()
print("Original Ssds")
for dataset in datasets:
    ssd = [s for s in ssds if s.name+".csv" == dataset.filename][0]
    print(ssd.id)


# =======================
#
# Step 8. Process graph patterns
#
# =======================

octo = sn.octopii.items[0]


alignment, _, _ = octo.get_alignment()
align_edges = alignment.edges(data=True, keys=True)
uri_lookup = dict()

for (_, _, key, data) in align_edges:
    uri_lookup[data["alignId"]] = key


embeds_path = project_path + "/stp/resources/museum_spt_pattern_bench/museum2_embeds_normal"
graph_j = os.path.join(embeds_path, "graphs.json")
edges_j = os.path.join(embeds_path, "edges.json")

patterns = []

print("---- processing edge lines!!!")
edge_lines = dict()
with open(edges_j) as fj:
    for line in fj:
        proc = json.loads(line)
        if proc["data"]["alignId"] in uri_lookup:
            edge_lines[proc["id"]] = uri_lookup[proc["data"]["alignId"]]
        else:
            print("Missing alignId: ", proc["data"]["alignId"])


print()
print("---- processing graph lines!!!")
graph_lines = []
with open(graph_j) as fj:
    for line in fj:
        proc = json.loads(line)
        el = proc['data']['__variable_mapping'].replace("}", "").replace("{", "").split(",")
        edges = []
        for e in el:
            pos = e.strip().split("=")
            if pos[0].startswith("__e"):
                if pos[1] in edge_lines:
                    edges.append(edge_lines[pos[1]])
                else:
                    print("     missing edge id: ", pos[1])

        graph_lines.append([proc['id'],
                            proc['data']['support'],
                            len(edges),
                            sorted(edges)])

graph_lines.sort(key=lambda x: x[3])

import csv
embeds_path = project_path + "/stp/resources/museum_spt_pattern_bench/museum2_embeds_normal_merged.csv"
with open(embeds_path, "w+") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["patter", "support", "num_edges", "edge_keys"])
    csv_writer.writerows(graph_lines)

# =======================
#
# Step 8. Use CP solver and compare!
#
# =======================

import stp.integration as integ
from stp.alignment import convert_ssd, to_graphviz, process_unknown
from importlib import reload

import stp.alignment as al

reload(integ)
reload(al)

print("Getting labels in the training set")
train_labels = []
for i in train_sample:
    train_labels += [d.full_label for d in new_ssds[i].data_nodes]
train_labels = set(train_labels)

print("Starting benchmark")
benchmark_results = [["octopus", "dataset", "ssd", "method", "solution", "status",
                      "jaccard", "precision", "recall",
                      "karma_jaccard", "karma_precision", "karma_recall",
                      "time"]]

folder = os.path.join(project_path, "stp", "resources", "unknown_run")
for dataset in datasets:
    ssd = [s for s in new_ssds if s.name+".csv" == dataset.filename][0]
    # we tell here the correct columns
    column_names = [c.name for c in ssd.columns]
    print("Chosen dataset: ", dataset)
    print("Chosen SSD id: ", ssd.id)

    # we should do prediction outside of solvers so that solvers use the cached results!
    print("Doing schema matcher part")
    df = octo.matcher_predict(dataset)

    cor_ssd = al.process_unknown(ssd, train_labels)

    # cp solver approach
    print("---> trying chuffed")
    try:
        integrat = integ.IntegrationGraph(octopus=octo, dataset=dataset,
                                          match_threshold=0.01, simplify_graph=True)
        solution = integrat.solve(column_names)
        runtime = time.time() - start
        # print(solution)
        if len(solution) < 1:
            logging.error("Chuffed found no solutions for ssd {}".format(ssd.id))
            raise Exception("Chuffed found no solutions!")
        for idx, sol in enumerate(solution):
            # convert solution to SSD
            cp_ssd = SSD().update(json.loads(sol), sn._datasets, sn._ontologies)
            eval = cp_ssd.evaluate(cor_ssd, False, True)
            comparison = sn.ssds.compare(cp_ssd, cor_ssd, False, False)
            res = [octo.id, dataset.id, ssd.id, "chuffed", idx, "success",
                   eval["jaccard"], eval["precision"], eval["recall"],
                   comparison["jaccard"], comparison["precision"], comparison["recall"],
                   runtime]
            benchmark_results.append(res)
    except Exception as e:
        print("CP solver failed to find solution: {}".format(e))
        res = [octo.id, dataset.id, ssd.id, "chuffed", 0, "fail", 0, 0, 0, 0, 0, 0, time.time() - start]
        benchmark_results.append(res)

    # karma approach
    print("---> trying karma")
    try:
        start = time.time()
        karma_ssd = octo.predict(dataset)[0].ssd
        runtime = time.time() - start
        eval = karma_ssd.evaluate(cor_ssd, False, True)
        comparison = sn.ssds.compare(karma_ssd, cor_ssd, False, False)
        res = [octo.id, dataset.id, ssd.id, "karma", 0, "success",
               eval["jaccard"], eval["precision"], eval["recall"],
               comparison["jaccard"], comparison["precision"], comparison["recall"],
               runtime]
    except Exception as e:
        print("Karma failed to find solution: {}".format(e))
        res = [octo.id, dataset.id, ssd.id, "karma", 0, "fail", 0, 0, 0, 0, 0, 0, 0]
    benchmark_results.append(res)


print("Finished benchmark")
import csv
result_csv = os.path.join(project_path, "stp", "resources", "unknown_benchmark_results_new.csv")
with open(result_csv, "w+") as f:
    csvf = csv.writer(f)
    csvf.writerows(benchmark_results)