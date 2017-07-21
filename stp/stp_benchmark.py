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
from serene.elements.semantics.base import KARMA_DEFAULT_NS

# from stp.alignment import read_karma_graph, add_matches, convert_ssd, to_graphviz
#
# from stp.integration import IntegrationGraph

project_path = "/home/natalia/PycharmProjects/serene-python-client"
benchmark_path = os.path.join(project_path, "tests", "resources", "museum_benchmark")
# benchmark_path = os.path.join(os.path.abspath(os.curdir), "tests", "resources", "museum_benchmark")
print("benchmark_path: ", benchmark_path)

import logging
from logging.handlers import RotatingFileHandler
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

input("Press enter to upload datasets and ssds...")

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


if len(datasets) != len(ssds):
    print("Something went wrong. Failed to read all datasets and ssds.")
    exit()

input("Press enter to continue...")
print("*********** Datasets and semantic source descriptions from museum benchmark")
ssd_path = "ssds/"
# for i, ds in enumerate(datasets):
#     print(i, ": dataset=", ds, "; ssd=", ssds[i])
#     # nx.write_edgelist(ssds[i].semantic_model._graph, os.path.join(ssd_path, ssds[i].name + ".edgelist"))
#     # nx.write_adjlist(ssds[i].semantic_model._graph, os.path.join(ssd_path, ssds[i].name + ".adjlist"))
#     # nx.write_multiline_adjlist(ssds[i].semantic_model._graph, os.path.join(ssd_path, ssds[i].name + ".madjlist"))
#     nx.write_graphml(ssds[i].semantic_model._graph, os.path.join(ssd_path, ssds[i].name + ".graphml"))

print()
ssds[0].show(outfile=ssds[0].name + ".ssd.dot")

input("Press enter to continue...")

# =======================
#
#  Step 4: Specify training and test samples
#
# =======================
train_sample = []
test_sample = []

# s16 will be test
for i, ssd in enumerate(ssds):
    # if "s07" in ssd.dataset.filename:
    #     test_sample.append(i)
    # else:
    #     # if "s07" in ds.filename or "s08" in ds.filename:
    #     train_sample.append(i)
    if i < 15:
        test_sample.append(i)
    else:
        train_sample.append(i)

# train_sample = train_sample[:5]

print("Indexes for training sample: ", train_sample)
print("Indexes for testing sample: ", test_sample)

# =======================
#
#  Step 5: Create octopus
#
# =======================

octo_local = sn.Octopus(
    ssds=[ssds[i] for i in train_sample],
    ontologies=ontologies,
    name='octopus-without-s07',
    description='Testing example for places and companies',
    resampling_strategy="NoResampling",  # optional
    num_bags=10,  # optional
    bag_size=30,  # optional
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
            ,"mean-character-cosine-similarity-from-class-examples"
            ,"min-editdistance-from-class-examples"
            ,"min-wordnet-jcn-distance-from-class-examples"
            ,"min-wordnet-lin-distance-from-class-examples"
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

if octo.state.status in {Status.ERROR}:
    print("Something went wrong. Failed to train the Octopus.")
    exit()

input("Press enter to continue...")

# =======================
#
# Step 7. Schema mapping using CP solver
#
# =======================
ssds = sn.ssds.items
octo = sn.octopii.items[0]
datasets = sn.datasets.items

import stp.integration as integ
from importlib import reload

dataset = [ds for ds in datasets if ds.id == 460826978][0]
ssd = [s for s in ssds if s.name+".csv" == dataset.filename][0]
column_names = [c.name for c in ssd.columns]
print("Chosen dataset: ", dataset)

reload(integ)
integrat = integ.IntegrationGraph(octopus=octo, dataset=dataset,
                               match_threshold=0.0, simplify_graph=True)
# fff = integrat._generate_oznfzn(column_names)

solution = integrat.solve(column_names, dot_file="{}.cpsol.dot".format(dataset.id))

print(solution)


# =======================
#
# Step 8. Generate benchmark for CP solver
#
# =======================

import stp.integration as integ
from stp.alignment import convert_ssd, to_graphviz
from importlib import reload

reload(integ)

print("Starting generation")

folder = os.path.join(project_path, "stp", "resources", "half2")
for dataset in datasets:
    ssd = [s for s in ssds if s.name+".csv" == dataset.filename][0]
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
                                      match_threshold=0.05, simplify_graph=True)

    # write alignment graph
    align_path = os.path.join(folder, "{}.alignment".format(ssd.id))
    nx.write_graphml(integrat.graph, align_path+".graphml")
    # to_graphviz(integrat.graph, out_file=align_path+".dot", prog="dot")
    print("---> alignment written")

    # write integration graph
    integration_graph = integrat._add_matches(column_names)
    integration_path = os.path.join(folder, "{}.integration".format(ssd.id))
    nx.write_graphml(integration_graph, integration_path + ".graphml")
    # to_graphviz(integration_graph, out_file=integration_path + ".dot", prog="dot")
    print("---> integration written")

    # write ground truth
    ssd_path = os.path.join(folder, "{}.ssd".format(ssd.id))
    ssd_nx = convert_ssd(ssd, integration_graph, ssd_path + ".graphml")
    to_graphviz(ssd_nx, out_file=ssd_path + ".dot", prog="dot")
    print("---> ssd written")


print("Finished generation")

# =======================
#
# Step 9. Compare chuffed and karma solutions!!!
#
# =======================

import stp.integration as integ
from importlib import reload

reload(integ)

print("Starting benchmark")
benchmark_results = [["octopus", "dataset", "ssd", "method", "solution", "status",
                      "jaccard", "precision", "recall",
                      "karma_jaccard", "karma_precision", "karma_recall",
                      "time"]]
for dataset in datasets:
    ssd = [s for s in ssds if s.name+".csv" == dataset.filename][0]
    # we tell here the correct columns
    column_names = [c.name for c in ssd.columns]
    print("Chosen dataset: ", dataset)

    # we should do prediction outside of solvers so that solvers use the cached results!
    print("Doing schema matcher part")
    df = octo.matcher_predict(dataset)

    # cp solver approach
    print("---> trying chuffed")
    start = time.time()
    try:
        integrat = integ.IntegrationGraph(octopus=octo, dataset=dataset,
                                          match_threshold=0.05, simplify_graph=True)
        solution = integrat.solve(column_names)
        runtime = time.time() - start
        # print(solution)
        if len(solution) < 1:
            raise Exception("Chuffed found no solutions!")
        for idx, sol in enumerate(solution):
            # convert solution to SSD
            cp_ssd = SSD().update(json.loads(sol), sn._datasets, sn._ontologies)
            eval = cp_ssd.evaluate(ssd, False, True)
            comparison = sn.ssds.compare(cp_ssd, ssd, False, False)
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
    start = time.time()
    try:
        karma_ssd = octo.predict(dataset)[0].ssd
        runtime = time.time() - start
        eval = karma_ssd.evaluate(ssd, False, True)
        comparison = sn.ssds.compare(karma_ssd, ssd, False, False)
        res = [octo.id, dataset.id, ssd.id, "karma", 0, "success",
               eval["jaccard"], eval["precision"], eval["recall"],
               comparison["jaccard"], comparison["precision"], comparison["recall"],
               runtime]
    except Exception as e:
        print("Karma failed to find solution: {}".format(e))
        res = [octo.id, dataset.id, ssd.id, "karma", 0, "fail", 0, 0, 0, 0, 0, 0, time.time() - start]
    benchmark_results.append(res)

print("Finished benchmark")
import csv
result_csv = os.path.join(project_path, "stp", "resources", "benchmark_results_tl20.csv")
with open(result_csv, "w+") as f:
    csvf = csv.writer(f)
    csvf.writerows(benchmark_results)

# # =======================
# #
# # Step 7. Predict
# #
# # =======================
#
# start = time.time()
# predicted = octo.predict(datasets[test_sample[0]])
# print("Prediction done in: {}".format(time.time() - start))
# print(predicted)
#
# print()
# print("Showing ground truth...")
# # ssds[test_sample[0]] is ground truth
# ground_truth = ssds[test_sample[0]]
# ground_truth.show(title='ground truth',
#                   outfile=os.path.join(tempfile.gettempdir(), 'ground_truth.png'))
# input("Press enter to see predicted semantic models...")
#
# print("><><><><")
# for res in predicted:
#     print(res)
#     print()
#     res.ssd.show()
#     input("Press enter to continue...")
# print("><><><><")
#
# # for p in predicted:
# #     print("Predicted candidate rank", p.score.rank)
# #     print("Score:")
# #     p.score.show()
# #     p.ssd.show()
# #     input("Press any key to continue...")
#
# # the best is number 0!
# predicted_ssd = predicted[0].ssd
# # predicted_ssd.unmapped_columns
# print()
# print("============Best recommendation=========")
# print(predicted_ssd)
#
# print("Mappings:")
# for map in predicted_ssd.mappings.items():
#     print(map)
#
# print("Unmapped columns:")
# print(predicted_ssd.unmapped_columns)
# print()
#
# print("Columns: ")
# for col in predicted_ssd.columns:
#     print("name={}, id={}".format(col.name, col.id))
#
# print()
# print(predicted_ssd.json)
# print()
# pprint(predicted_ssd.json)
# input("Press enter to continue...")
# # =======================
# #
# # Step 8. Evaluate against ground truth
# #
# # =======================
#
# # ssds[test_sample[0]] is ground truth
# ground_truth = ssds[test_sample[0]]
# comparison = sn.ssds.compare(predicted_ssd, ground_truth, False, False)
# predicted_ssd.show(title="best recommendation: \n"+str(comparison),
#                    outfile=os.path.join(tempfile.gettempdir(), 'best_recommendation.png'))
# # ground_truth.show(title='ground truth',
# #                   outfile=os.path.join(tempfile.gettempdir(), 'ground_truth.png'))
# print("================")
#
# input("Press enter to continue...")
# for i, pred in enumerate(predicted):
#     comparison = sn.ssds.compare(pred.ssd, ssds[test_sample[0]], False, False)
#     print("SsdResult({}) comparison: {}".format(i,comparison))
