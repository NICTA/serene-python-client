# ===========
#    SETUP
# ===========
try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene

import datetime
import pandas as pd
import os
import time

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

# =======================
#
#  Step 3: Add datasets from the museum benchmark and corresponding semantic source descriptions
#
# =======================
dataset_dir = os.path.join(benchmark_path, "dataset")
ssd_dir = os.path.join(benchmark_path, "ssd")

datasets = [] # list of museum datasets
ssds = [] # list of semantic source descriptions from museum benchmark

for ds in os.listdir(dataset_dir):
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

print("*********** Datasets and semantic source descriptions from museum benchmark")
for i, ds in enumerate(datasets):
    print(i, ": dataset=", ds, "; ssd=", ssds[i])

# =======================
#
#  Step 4: Specify training and test samples
#
# =======================
train_sample = []
test_sample = []

# s16 will be test
for i, ds in enumerate(datasets):
    if "s16" in ds.filename:
        test_sample.append(i)
    else:
        train_sample.append(i)

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
    name='octopus-without-s16',
    description='Testing example for places and companies',
    resampling_strategy="NoResampling",  # optional
    num_bags=100,  # optional
    bag_size=10,  # optional
    model_type="randomForest",
    modeling_props={

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
            "entropy-for-discrete-values"
        ],
        "activeFeatureGroups": [
            "inferred-data-type",
            "stats-of-text-length",
            "stats-of-numeric-type",
            "prop-instances-per-class-in-knearestneighbours",
            "mean-character-cosine-similarity-from-class-examples",
            "min-editdistance-from-class-examples",
            "min-wordnet-jcn-distance-from-class-examples",
            "min-wordnet-lin-distance-from-class-examples"
        ],
        "featureExtractorParams": [
            {
                "name": "prop-instances-per-class-in-knearestneighbours",
                "num-neighbours": 3
            }, {
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

# =======================
#
# Step 7. Predict
#
# =======================

start = time.time()
predicted = octo.predict(datasets[test_sample[0]])
print("Prediction done in: {}".format(time.time() - start))

print("><><><><")
for k in predicted['predictions']:
    print(k['score'])
    print()
    SSD().update(k['ssd'], sn.datasets, sn.ontologies).show()
    input("Press enter to continue...")
print("<<<<<<<<")

print(predicted)

# for p in predicted:
#     print("Predicted candidate rank", p.score.rank)
#     print("Score:")
#     p.score.show()
#     p.ssd.show()
#     input("Press any key to continue...")

# the best is number 0!
predicted_ssd = predicted[0].ssd

# =======================
#
# Step 8. Evaluate against ground truth
#
# =======================