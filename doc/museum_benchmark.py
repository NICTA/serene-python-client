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

from serene import SSD, Status, DataProperty, Mapping, ObjectProperty, Column, Class, DataNode, ClassNode
from serene.elements.semantics.base import KARMA_DEFAULT_NS


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

benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "museum_benchmark")
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
for i, ds in enumerate(datasets):
    print(i, ": dataset=", ds, "; ssd=", ssds[i])
print()

# =======================
#
#  Step 4: Specify training and test samples
#
# =======================
train_sample = []
test_sample = []

# s16 will be test
for i, ds in enumerate(datasets):
    if "s07" in ds.filename:
        test_sample.append(i)
    else:
        # if "s07" in ds.filename or "s08" in ds.filename:
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
    resampling_strategy="BaggingToMean",  # optional
    num_bags=10,  # optional
    bag_size=30,  # optional
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
        ],
        "activeFeatureGroups": [
            "char-dist-features",
            "inferred-data-type",
            "stats-of-text-length",
            "stats-of-numeric-type"
            ,"prop-instances-per-class-in-knearestneighbours"
            ,"mean-character-cosine-similarity-from-class-examples"
            # ,"min-editdistance-from-class-examples"
            # ,"min-wordnet-jcn-distance-from-class-examples"
            # ,"min-wordnet-lin-distance-from-class-examples"
        ],
        "featureExtractorParams": [
            {
                "name": "prop-instances-per-class-in-knearestneighbours",
                "num-neighbours": 3
            }
        # , {
        #         "name": "min-editdistance-from-class-examples",
        #         "max-comparisons-per-class": 3
        #     }, {
        #         "name": "min-wordnet-jcn-distance-from-class-examples",
        #         "max-comparisons-per-class": 3
        #     }, {
        #         "name": "min-wordnet-lin-distance-from-class-examples",
        #         "max-comparisons-per-class": 3
        #     }
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

# =======================
#
# Step 7. Predict
#
# =======================

start = time.time()
predicted = octo.predict(datasets[test_sample[0]])
print("Prediction done in: {}".format(time.time() - start))
print(predicted)

print()
print("Showing ground truth...")
# ssds[test_sample[0]] is ground truth
ground_truth = ssds[test_sample[0]]
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
print("============Best recommendation=========")
print(predicted_ssd)

print("Mappings:")
for map in predicted_ssd.mappings.items():
    print(map)

print("Unmapped columns:")
print(predicted_ssd.unmapped_columns)
print()

print("Columns: ")
for col in predicted_ssd.columns:
    print("name={}, id={}".format(col.name, col.id))

print()
print(predicted_ssd.json)
print()
pprint(predicted_ssd.json)
input("Press enter to continue...")
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
    comparison = sn.ssds.compare(pred.ssd, ssds[test_sample[0]], False, False)
    print("SsdResult({}) comparison: {}".format(i,comparison))