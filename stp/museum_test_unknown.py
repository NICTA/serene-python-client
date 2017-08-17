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

benchmark_path = os.path.join(os.environ["SERENE_BENCH"], "museum_benchmark")

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

print("*********** Ontologies from museum benchmark")
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


import multiprocessing as mp
from functools import partial
import time
import json
import pandas as pd

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


def do_chuffed(server, ch_dataset, ch_column_names, ch_ssd, orig_ssd,
               train_flag=False, chuffed_path=None, simplify=True,
               patterns=None, experiment="loo", soft_assumptions=False,
               pattern_sign=1.0, accu=0.0, pattern_time=None):
    res = []
    start = time.time()
    try:
        integrat = integ.IntegrationGraph(octopus=my_octopus, dataset=ch_dataset,
                                          match_threshold=0.001, simplify_graph=simplify,
                                          patterns=patterns, pattern_sign=pattern_sign,
                                          alignment_graph=octopus_graph,
                                          alignment_node_map=octopus_node_map,
                                          alignment_link_map=octopus_link_map)
        if chuffed_path:
            integrat._chuffed_path = chuffed_path  # change chuffed solver version
        solution = integrat.solve(ch_column_names,
                                  soft_assumptions=soft_assumptions, relax_every=2)
        runtime = time.time() - start
        # get cost of the ground truth
        try:
            nx_ssd = al.convert_ssd(ch_ssd, integrat.graph, file_name=None, clean_ssd=False)
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

        for idx, sol in enumerate(solution):
            # convert solution to SSD
            json_sol = json.loads(sol)
            logging.info("Converted solution: {}".format(json_sol))
            cp_ssd = SSD().update(json_sol, server._datasets, server._ontologies)
            eval = cp_ssd.evaluate(ch_ssd, False, True)

            chuffed_time = json_sol["time"]
            match_score = json_sol["match_score"]
            cost = json_sol["cost"]
            objective = json_sol["objective"]

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

            res += [[experiment, my_octopus.id, ch_dataset.id, orig_ssd.name,
                     orig_ssd.id, "chuffed", idx, "success",
                     eval["jaccard"], eval["precision"], eval["recall"],
                     comparison["jaccard"], comparison["precision"], comparison["recall"],
                     train_flag, runtime, chuffed_path, simplify, chuffed_time,
                     soft_assumptions, pattern_sign, accu, sol_accuracy,
                     match_score, cost, objective,
                     cor_match_score, cor_cost, pattern_time,
                     integrat.graph.number_of_nodes(), integrat.graph.number_of_edges()]]
    except Exception as e:
        print("CP solver failed to find solution: {}".format(e))
        res += [[experiment, my_octopus.id, ch_dataset.id, orig_ssd.name, orig_ssd.id,
                 "chuffed", 0, "fail", None, None, None, None, None, None,
                 train_flag, time.time() - start, chuffed_path, simplify, None,
                 soft_assumptions, pattern_sign, accu, None,
                 None, None, None, None, None, pattern_time, None, None]]
    return res


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copyreg
import types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# =======================
#
#  Karma style: Changing # known semantic models
#
# =======================
def simple_chuffed(server=None, ch_dataset=None,
                   ch_column_names=None, ch_ssd=None, orig_ssd=None,
                   train_flag=False, chuffed_path=None, simplify=True,
                   patterns=None, experiment="loo", soft_assumptions=False,
                   pattern_sign=1.0, accu=0.0, pattern_time=None):
    print("HERE: ", soft_assumptions)
    return []

def start_chuffed():
    # cp solver approach
    funclist = []
    print("OCtopus type: {}".format(my_octopus))
    func = partial(do_chuffed
                   , server=sn
                   , ch_dataset=dataset
                   , ch_column_names=column_names,
                   ch_ssd=cor_ssd,
                   orig_ssd=ssd,
                   train_flag=len(train_sample),
                   accu=sm_accuracy, experiment="eknown_{}".format(len(train_sample)),
                   pattern_time=pattern_time)

    for (chuffed, simpl, pat, soft, pts) in chuffed_paths:
        print("---> trying chuffed {}".format(chuffed))

        partf = pool.apply_async(func, kwds={"simplify": simpl, "patterns": pts,
                                             "soft_assumptions": soft,
                                             "pattern_sign": pts})
        funclist.append(partf)
        # res = al.do_chuffed(sn, octo, dataset, column_names, cor_ssd, ssd,
        #                     train_flag=False,
        #                     chuffed_path=chuffed, simplify=simpl,
        #                     patterns=pat, soft_assumptions=soft,
        #                     pattern_sign=pts, accu=sm_accuracy,
        #                     experiment="eknown_{}".format(len(train_sample)))

    pool.close()
    pool.join()
    results = []
    print("Trying to get results...")
    for partff in funclist:
        ans = partff.get()
        csvf.writerows(ans)
        results.append(ans)  # timeout in 100 seconds

    return results

if __name__ == "__main__":
    chuffed_lookup_path = os.path.join(project_path, "stp", "minizinc")
    result_csv = os.path.join(project_path, "stp", "resources", "unknown_benchmark_test.csv")

    NUM_PROC = mp.cpu_count() - 2

    openf = open(result_csv, "w+")

    csvf = csv.writer(openf)
    csvf.writerow(["experiment", "octopus", "dataset", "name", "ssd",
                   "method", "solution", "status",
                   "jaccard", "precision", "recall",
                   "karma_jaccard", "karma_precision", "karma_recall",
                   "train_flag", "time", "chuffed_path", "simplify", "chuffed_time",
                   "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                   "match_score", "cost", "objective", "cor_match_score", "cor_cost", "pattern_time",
                   "integration_nodes", "integration_links"])  # header

    sample_range = list(range(len(new_ssds)))
    len_sample = len(sample_range)
    benchmark_results = [["experiment", "octopus", "dataset", "name", "ssd",
                          "method", "solution", "status",
                          "jaccard", "precision", "recall",
                          "karma_jaccard", "karma_precision", "karma_recall",
                          "train_flag", "time", "chuffed_path", "simplify", "chuffed_time",
                          "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                          "match_score", "cost", "objective", "cor_match_score", "cor_cost", "pattern_time",
                          "integration_nodes", "integration_links"]]

    print("Starting benchmark")

    for cur_id in [0]: #sample_range:
        print("Currently selected ssd: ", cur_id)
        test_sample = [cur_id]

        for num in [5]: #range(1, len(sample_range)-1):
            train_sample = sample_range[:max(cur_id + num + 1 - len_sample, 0)] + sample_range[
                                                                                  cur_id + 1: cur_id + 1 + num]
            print("     train sample size: ", num)

            try:
                my_octopus = al.create_octopus(sn, new_ssds, train_sample, ontologies)
            except Exception as e:
                logging.error("Octopus creation failed: {}".format(e))
                print("Octopus creation failed: {}".format(e))
                continue

            octo_csv = os.path.join(project_path, "stp", "resources", "storage",
                                    "patterns.{}.csv".format(my_octopus.id))
            pattern_time = None
            # try:
            #     octo_patterns = my_octopus.get_patterns(octo_csv)
            #     start_time = time.time()
            #     pattern_time = time.time() - start_time
            # except Exception as e:
            #     print("failed to get patterns: {}".format(e))
            #     logging.warning("failed to get patterns: {}".format(e))
            #     octo_patterns = None
            octo_patterns= None

            chuffed_paths = [
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170718"),
                 True, None, False, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170722"),
                 True, None, False, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170722"),
                 True, None, True, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, False, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 1.0),
                (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
                 False, octo_patterns, True, 5.0)]

            print("Getting labels in the training set")
            train_labels = []
            for i in train_sample:
                train_labels += [d.full_label for d in new_ssds[i].data_nodes]
            train_labels = set(train_labels)

            ssd = new_ssds[cur_id]
            my_dataset = [ds for ds in datasets if ds.filename == ssd.name + ".csv"][0]
            # we tell here the correct columns
            column_names = [c.name for c in ssd.columns]
            print("Chosen dataset: ", my_dataset)
            print("Chosen SSD id: ", ssd.id)

            # we should do prediction outside of solvers so that solvers use the cached results!
            print("Doing schema matcher part")
            octopus_predicted_df = my_octopus.matcher_predict(my_dataset)
            cor_ssd = al.process_unknown(ssd, train_labels)
            try:
                sm_accuracy = al.compare_semantic_label(al.semantic_label_df(cor_ssd, "user_label"),
                                                 octopus_predicted_df[["column_id", "label"]])["accuracy"]
            except Exception as e:
                logging.error("Match score fail: {}".format(e))
                print("Match score fail: {}".format(e))
                sm_accuracy = None

            octopus_graph, octopus_node_map, octopus_link_map = my_octopus.get_alignment()
            # cp solver approach
            print("****Starting pool of workers...")
            pool = mp.Pool(NUM_PROC)

            start_chuffed()
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

            # karma approach
            print("---> trying karma")
            res = al.do_karma(sn, my_octopus, my_dataset, cor_ssd, ssd, train_flag=False,
                              accu=sm_accuracy, experiment="eknown_{}".format(len(train_sample)))
            benchmark_results += res
            csvf.writerows(res)
            openf.flush()

    print("Benchmark finished!")
    print(benchmark_results)

    openf.close()