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
import tarfile
from importlib import reload
import csv

if "SERENE_PYTHON" not in os.environ:
    os.environ["SERENE_PYTHON"] = "/home/natalia/PycharmProjects/serene-python-client"

project_path = os.environ["SERENE_PYTHON"]

from serene import SSD
from serene.elements.semantics.base import KARMA_DEFAULT_NS, DEFAULT_NS, ALL_CN, OBJ_PROP, UNKNOWN_CN, UNKNOWN_DN

try:
    import integration as integ
    import alignment as al
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import integration as integ
    import alignment as al

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
#  Start with a connection to the server...
#
# =======================
sn = serene.Serene(
    host='127.0.0.1',
    port=8080,
)
print(sn)

if "SERENE_BENCH" not in os.environ:
    os.environ["SERENE_BENCH"] = os.path.join(project_path, "tests", "resources")

benchmark_path = os.path.join(os.environ["SERENE_BENCH"], "soccer")

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
#  Add ontologies from the museum benchmark
#
# =======================
owl_dir = os.path.join(benchmark_path, "owl")

ontologies = []
for path in os.listdir(owl_dir):
    f = os.path.join(owl_dir, path)
    ontologies.append(sn.ontologies.upload(f))

# add Unknown class
# unknown_owl = os.path.join("unknown", "unknown_class.ttl")
unknown_owl = os.path.join(project_path, "stp", "unknown", "unknown_class.ttl")
print("Unknown ontology ", unknown_owl)
ontologies.append(sn.ontologies.upload(unknown_owl))

print("*********** Ontologies from soccer benchmark")
for onto in ontologies:
    print(onto)
print()

# =======================
#
#  Add datasets from the museum benchmark and corresponding semantic source descriptions
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
    # ds_name = os.path.splitext(ds)[0]
    ds_name = ds
    # associated semantic source description for this dataset
    ssd_f = os.path.join(ssd_dir, ds_name + ".ssd")

    print("Adding dataset: " + str(ds_f))
    dataset = sn.datasets.upload(ds_f, description="soccer_benchmark")
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
#  Add unknown mappings to ssds
#
# ======================
from copy import deepcopy

new_ssds = [deepcopy(s) for s in ssds]

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
#  Helper methods
#
# =======================
reload(integ)
reload(al)
reload(serene)


def do_experiment(octopus, orig_ds, orig_ssd, avail_labels, chosen_paths, chosen_columns,
                  experiment_descr, benchmark_results, csvfile, known_num, pattern_time):
    """

    :param octopus:
    :param orig_ds:
    :param orig_ssd:
    :param avail_labels:
    :param chosen_paths:
    :param chosen_columns:
    :param more_ssd:
    :param experiment_descr:
    :param benchmark_results:
    :param csvfile:
    :return:
    """
    # we should do prediction outside of solvers so that solvers use the cached results!
    print("Doing schema matcher part")
    df = octopus.matcher_predict(orig_ds)
    cor_ssd = al.process_unknown(orig_ssd, avail_labels)
    try:
        sm_accuracy = al.compare_semantic_label(al.semantic_label_df(cor_ssd, "user_label"),
                                                df[["column_id", "label"]])["accuracy"]
    except Exception as e:
        logging.error("Match score fail: {}".format(e))
        print("Match score fail: {}".format(e))
        sm_accuracy = None

    # cp solver approach
    for (chuffed, simpl, pat, soft, pts) in chosen_paths:
        print("---> trying chuffed {}".format(chuffed))
        res = al.do_chuffed(sn, octopus, orig_ds, chosen_columns, cor_ssd, ssd,
                            train_flag=known_num,
                            chuffed_path=chuffed, simplify=simpl,
                            patterns=pat, soft_assumptions=soft,
                            pattern_sign=pts, accu=sm_accuracy,
                            experiment=experiment_descr, pattern_time=pattern_time)
        benchmark_results += res
        csvfile.writerows(res)

    # karma approach
    print("---> trying karma")
    res = al.do_karma(sn, octopus, orig_ds, cor_ssd, ssd, train_flag=False,
                      accu=sm_accuracy, experiment=experiment_descr, pattern_time=pattern_time)
    benchmark_results += res
    csvfile.writerows(res)
# =======================
#
#  Karma style: Changing # known semantic models
#
# =======================
import time
import yaml
chuffed_lookup_path = os.path.join(project_path, "stp", "minizinc")
result_csv = os.path.join(project_path, "stp", "resources", "results", "soccer_test.csv")
with open(result_csv, "w+") as f:
    csvf = csv.writer(f)
    csvf.writerow(["experiment", "octopus", "dataset", "name", "ssd",
                   "method", "solution", "status",
                   "jaccard", "precision", "recall",
                   "karma_jaccard", "karma_precision", "karma_recall",
                   "known_num", "time", "chuffed_path", "simplify", "chuffed_time",
                   "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                   "match_score", "cost", "objective", "cor_match_score", "cor_cost",
                   "pattern_time", "number_nodes", "number_links", "extended_time"])  # header

    sample_range = list(range(len(new_ssds)))
    len_sample = len(sample_range)
    benchmark_results = [["experiment", "octopus", "dataset", "name", "ssd",
                          "method", "solution", "status",
                          "jaccard", "precision", "recall",
                          "karma_jaccard", "karma_precision", "karma_recall",
                          "train_flag", "time", "chuffed_path", "simplify", "chuffed_time",
                          "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                          "match_score", "cost", "objective", "cor_match_score", "cor_cost", "pattern_time",
                          "extended_time"]]

    print("Starting benchmark")

    for cur_id in [0]:#sample_range:
        print("Currently selected ssd: ", cur_id)
        test_sample = [cur_id]

        for num in [5]: #range(1, len(sample_range)):
            train_sample = sample_range[:max(cur_id + num + 1 - len_sample, 0)] + sample_range[cur_id + 1: cur_id + 1 + num]
            print("     train sample size: ", num)

            try:
                octo = al.create_octopus(sn, new_ssds, train_sample, ontologies)
            except Exception as e:
                logging.error("Octopus creation failed: {}".format(e))
                print("Octopus creation failed: {}".format(e))
                continue

            # try:
            #     octo2 = al.create_octopus2(sn, new_ssds, train_sample, ontologies)
            # except Exception as e:
            #     logging.error("Octopus2 creation failed: {}".format(e))
            #     octo2 = None
            #
            # try:
            #     octo3 = al.create_octopus3(sn, new_ssds, train_sample, ontologies)
            # except Exception as e:
            #     logging.error("Octopus3 creation failed: {}".format(e))
            #     octo3 = None

            octo_csv = os.path.join(project_path, "stp", "resources", "storage",
                                    "patterns.{}.csv".format(octo.id))
            start_time = time.time()
            pattern_time = None
            try:
                with open(os.path.join(project_path, "stp", 'config.yaml'), 'r') as stream:
                    config = yaml.load(stream)
                print("Serene-path: ", config["embeds-path"])
                octo_patterns = octo.get_patterns(octo_csv, serene_path=str(config["embeds-path"]))
                pattern_time = time.time() - start_time
                print("Patterns obtained")
            except Exception as e:
                print("failed to get patterns: {}".format(e))
                logging.warning("failed to get patterns: {}".format(e))
                octo_patterns = None
            print("Uploaded octopus:", octo)

            chuffed_paths = [
                # (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170718"),
                #  True, None, False, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170722"),
                 True, None, True, 1.0),
                # (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                #  False, octo_patterns, False, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 5.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 10.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 20.0),
            ]

            print("Getting labels in the training set")
            train_labels = []
            for i in train_sample:
                train_labels += [d.full_label for d in new_ssds[i].data_nodes]
            train_labels = set(train_labels)

            ssd = new_ssds[cur_id]
            dataset = [ds for ds in datasets if ds.filename == ssd.name][0]
            # we tell here the correct columns
            column_names = [c.name for c in ssd.columns]
            print("Chosen dataset: ", dataset)
            print("Chosen SSD id: ", ssd.id)

            do_experiment(octopus=octo, orig_ds=dataset, orig_ssd=ssd, avail_labels=train_labels,
                          chosen_paths=chuffed_paths, chosen_columns=column_names,
                          experiment_descr="octo1_known_{}".format(len(train_sample)),
                          benchmark_results=benchmark_results, csvfile=csvf, known_num=len(train_sample),
                          pattern_time=pattern_time)

            # if octo2:
            #     do_experiment(octopus=octo2, orig_ds=dataset, orig_ssd=ssd, avail_labels=train_labels,
            #                   chosen_paths=chuffed_paths, chosen_columns=column_names,
            #                   experiment_descr="octo2_known_{}".format(len(train_sample)),
            #                   benchmark_results=benchmark_results, csvfile=csvf, known_num=len(train_sample),
            #               pattern_time=pattern_time)
            # if octo3:
            #     do_experiment(octopus=octo3, orig_ds=dataset, orig_ssd=ssd, avail_labels=train_labels,
            #                   chosen_paths=chuffed_paths, chosen_columns=column_names,
            #                   experiment_descr="octo3_known_{}".format(len(train_sample)),
            #                   benchmark_results=benchmark_results, csvfile=csvf, known_num=len(train_sample),
            #               pattern_time=pattern_time)
            # we should do prediction outside of solvers so that solvers use the cached results!
            # print("Doing schema matcher part")
            # df = octo.matcher_predict(dataset)
            # cor_ssd = al.process_unknown(ssd, train_labels)
            # try:
            #     sm_accuracy = al.compare_semantic_label(al.semantic_label_df(cor_ssd, "user_label"),
            #                                      df[["column_id", "label"]])["accuracy"]
            # except Exception as e:
            #     logging.error("Match score fail: {}".format(e))
            #     print("Match score fail: {}".format(e))
            #     sm_accuracy = None

            # # cp solver approach
            # for (chuffed, simpl, pat, soft, pts) in chuffed_paths:
            #     print("---> trying chuffed {}".format(chuffed))
            #     res = al.do_chuffed(sn, octo, dataset, column_names, cor_ssd, ssd,
            #                         train_flag=False,
            #                         chuffed_path=chuffed, simplify=simpl,
            #                         patterns=pat, soft_assumptions=soft,
            #                         pattern_sign=pts, accu=sm_accuracy,
            #                         experiment="eknown_{}".format(len(train_sample)))
            #     benchmark_results += res
            #     csvf.writerows(res)
            #
            # # karma approach
            # print("---> trying karma")
            # res = al.do_karma(sn, octo, dataset, cor_ssd, ssd, train_flag=False,
            #                   accu=sm_accuracy, experiment="eknown_{}".format(len(train_sample)))
            # benchmark_results += res
            # csvf.writerows(res)
            f.flush()

    print("Benchmark finished!")
    print(benchmark_results)