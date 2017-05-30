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

from serene import SSD, Status, DataProperty, Mapping, ObjectProperty, Column, Class, DataNode, ClassNode
from serene.elements import SubClassLink, ObjectLink, ColumnLink
from serene.elements.semantics.base import KARMA_DEFAULT_NS, DEFAULT_NS, ALL_CN, OBJ_PROP

from stp.alignment import read_karma_graph, add_matches, convert_ssd, to_graphviz, convert_karma_graph
from collections import defaultdict

import logging
from logging.handlers import RotatingFileHandler
# setting up the logging
log_file = os.path.join("resources", 'benchmark_stp.log')
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

# benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "museum_benchmark")
benchmark_path = os.path.join(os.path.abspath(os.curdir), "tests", "resources", "museum_benchmark")
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
#  Step 5: Add unknown mappings to ssds
#
# ======================
from copy import deepcopy

def fill_unknown(original_ssd):
    """
    Make all unmapped columns from the dataset as instances of unknown.
    Then add Thing node and make all class nodes as subclass of Thing.
    This way we ensure that ssd is connected
    """
    s = original_ssd  # maybe make a deep copy here?
    unmapped_cols = [col for col in s.dataset.columns if col not in s.columns]
    for i, col in enumerate(unmapped_cols):
        s._semantic_model.add_node(col, index=col.id)
        data_node = DataNode(ClassNode("Unknown", prefix="http://au.csiro.data61/serene/dev#"), label="unknown",
                             index=i,
                             prefix="http://au.csiro.data61/serene/dev#")
        s.map(col, data_node)
    if len(unmapped_cols):
        s = add_all_node(s)
    return s

def add_all_node(original_ssd):
    """Add All node to ssd and make all class nodes connect to All"""
    ssd = original_ssd
    class_nodes = ssd.class_nodes
    prefix = "http://au.csiro.data61/serene/dev#"
    thing_node = ClassNode("All", prefix=prefix)
    ssd._semantic_model.add_node(thing_node)
    for cn in class_nodes:
        ssd._semantic_model.add_edge(
            thing_node, cn, ObjectLink("connect", prefix=prefix))
    return ssd

def add_thing_node(original_ssd):
    """Add Thing node to ssd and make all class nodes subclass of Thing"""
    ssd = original_ssd
    class_nodes = ssd.class_nodes
    prefix = "http://www.w3.org/2000/01/rdf-schema#"
    thing_node = ClassNode("Thing", prefix="http://www.w3.org/2002/07/owl#")
    ssd._semantic_model.add_node(thing_node)
    for cn in class_nodes:
        ssd._semantic_model.add_edge(
            cn, thing_node, SubClassLink("subClassOf", prefix=prefix))
    return ssd

new_ssds = [deepcopy(s) for s in sn.ssds.items]

for s in new_ssds:
    s = fill_unknown(s)
    # s.show()
    # input("Press enter to continue...")


# upload ssds with unknown mappings
new_ssds = [sn.ssds.upload(s) for s in new_ssds]
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
    if i < 15:
        test_sample.append(i)
    else:
        train_sample.append(i)

# train_sample = train_sample[:5]

print("Indexes for training sample: ", train_sample)
print("Indexes for testing sample: ", test_sample)

# =======================
#
#  Step 7: Create octopus
#
# =======================

octo_local = sn.Octopus(
    ssds=[new_ssds[i] for i in train_sample],
    ontologies=ontologies,
    name='octopus-without-s07',
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
# Step 7'. Matcher Predict and construct the integration per each ssd!
#
# =======================

align_num = "test"
# get alignment graph as nx.MultiDiGraph
print("converting alignment graph json")
alignment = octo.get_alignment()
alignment_path = os.path.join("stp","resources", align_num, "alignment.graphml")
# write alignment graph to the disk
nx.write_graphml(alignment, alignment_path)
print("graphviz alignment")
_ = to_graphviz(alignment, os.path.join("stp","resources", align_num, "alignment.dot"), prog="dot")

print("showing predictions for the dataset...")
for i, ds in enumerate(datasets):
    print("predicting dataset: ", ds.filename)
    pred_df = octo.matcher_predict(ds)
    print("prediction finished")
    pred_df.to_csv(os.path.join("resources", align_num, "matches", ds.filename+".matches.csv"), index=False)

    # read matches
    pred_df = pd.read_csv(os.path.join("resources", align_num, "matches", ds.filename+".matches.csv"))

    # construct integration graph by adding matches for the columns
    column_names = [c.name for c in ssds[i].columns]
    integration = add_matches(alignment, pred_df, column_names, threshold=0.02)
    out_file = os.path.join("resources", align_num, str(i)+".integration.graphml")
    print("Writing integration graph to file {}".format(out_file))
    nx.write_graphml(integration, out_file)

    print("graphviz alignment")
    to_graphviz(integration, os.path.join("resources", align_num, str(i) + ".integration.dot"), prog="neato")

    # write ground truth
    out_file = os.path.join("resources", align_num, str(i) + ".ssd.graphml")
    ssd_graph = convert_ssd(ssds[i], integration, out_file)
    to_graphviz(ssd_graph, os.path.join("resources", align_num, str(i) + ".ssd.dot"))

input("Press enter to continue...")

# =======================
#
# Step 7. Predict
#
# =======================

start = time.time()
ds = [d for d in datasets if new_ssds[test_sample[0]].dataset.filename == d.filename][0]
predicted = octo.predict(ds)
print("Prediction done in: {}".format(time.time() - start))
print(predicted)

print()
print("Showing ground truth...")
# ssds[test_sample[0]] is ground truth
ground_truth = new_ssds[test_sample[0]]
ground_truth.show(title='ground truth',
                  outfile=os.path.join(tempfile.gettempdir(), 'ground_truth.png'))
input("Press enter to see predicted semantic models...")

print("><><><><")
for res in predicted:
    print(res)
    print()
    res.ssd.show()
    input("Press enter to continue...")
print("><><><><")

# for p in predicted:
#     print("Predicted candidate rank", p.score.rank)
#     print("Score:")
#     p.score.show()
#     p.ssd.show()
#     input("Press any key to continue...")

# the best is number 0!
predicted_ssd = predicted[0].ssd
# predicted_ssd.unmapped_columns
print()

# =======================
#
# Step 8. Evaluate against ground truth
#
# =======================

# ssds[test_sample[0]] is ground truth
ground_truth = ssds[test_sample[0]]
comparison = sn.ssds.compare(predicted_ssd, ground_truth, False, False)
predicted_ssd.show(title="best recommendation: \n"+str(comparison),
                   outfile=os.path.join(tempfile.gettempdir(), 'best_recommendation.png'))
# ground_truth.show(title='ground truth',
#                   outfile=os.path.join(tempfile.gettempdir(), 'ground_truth.png'))
print("================")

input("Press enter to continue...")
for i, pred in enumerate(predicted):
    comparison = sn.ssds.compare(pred.ssd, ground_truth, False, False)
    print("SsdResult({}) comparison: {}".format(i,comparison))
