"""This is a test script of model evaluation module."""
import os.path

import unittest2 as unittest
from pandas import read_csv, concat

from serene.matcher.eval import Column
from serene.matcher.eval import ModelEvaluation, Dataset, Model


class TestEval(unittest.TestCase):
    """

    """
    def test_eval(self):
        raise NotImplementedError("Test not implemented")
    # """
    # Tests the evaluation code...
    # """
    # DATA_PATH = "tests/resources"
    #
    # def _get_labels(self, datasets):
    #     """
    #     Returns a Column label map for the datasets...
    #     """
    #
    #     labels = read_csv(os.path.join(self.DATA_PATH, 'labels.csv'))
    #
    #     column_dataset_series = labels['attr_id']
    #
    #     dataset_series = column_dataset_series \
    #         .map(lambda column_dataset: column_dataset.rpartition('@')[2]) \
    #         .map(lambda dataset: dataset[:-4])
    #
    #     column_series = [c.rpartition('@')[0] for c in column_dataset_series]
    #
    #     labels = concat([dataset_series, column_series, labels['class']], axis=1)
    #     labels.columns = ['dataset', 'column', 'label']
    #
    #     labels = labels.to_records(index=False)
    #
    #     datasets = {dataset.description: dataset for dataset in datasets}
    #
    #     return {
    #         Column(name=column, dataset=datasets[dataset]): label
    #         for (dataset, column, label) in labels
    #         if dataset in datasets
    #     }
    #
    # def test_eval(self):
    #     """
    #     UnitTest for the evaluation model
    #     """
    #     model = Model(
    #         description='evaluation model',
    #         features={
    #             "activeFeatures": [
    #                 "num-unique-vals",
    #                 "prop-unique-vals",
    #                 "prop-missing-vals",
    #                 "ratio-alpha-chars",
    #                 "prop-numerical-chars",
    #                 "prop-whitespace-chars",
    #                 "prop-entries-with-at-sign",
    #                 "prop-entries-with-hyphen",
    #                 "prop-range-format",
    #                 "is-discrete",
    #                 "entropy-for-discrete-values"
    #             ],
    #             "activeFeatureGroups": [
    #                 "inferred-data-type",
    #                 "stats-of-text-length",
    #                 "stats-of-numeric-type",
    #                 "prop-instances-per-class-in-knearestneighbours",
    #                 "mean-character-cosine-similarity-from-class-examples",
    #                 "min-editdistance-from-class-examples",
    #                 "min-wordnet-jcn-distance-from-class-examples",
    #                 "min-wordnet-lin-distance-from-class-examples"
    #             ],
    #             "featureExtractorParams": [
    #                 {
    #                     "name": "prop-instances-per-class-in-knearestneighbours",
    #                     "num-neighbours": 3
    #                 }, {
    #                     "name": "min-editdistance-from-class-examples",
    #                     "max-comparisons-per-class": 20
    #                 }, {
    #                     "name": "min-wordnet-jcn-distance-from-class-examples",
    #                     "max-comparisons-per-class": 20
    #                 }, {
    #                     "name": "min-wordnet-lin-distance-from-class-examples",
    #                     "max-comparisons-per-class": 20
    #                 }
    #             ]
    #         },
    #         resampling_strategy="ResampleToMean"
    #     )
    #
    #     datasets = [
    #         Dataset(
    #             description='homeseekers',
    #             file_path=os.path.join(self.DATA_PATH, 'homeseekers.csv')
    #         ),
    #         Dataset(
    #             description='yahoo',
    #             file_path=os.path.join(self.DATA_PATH, 'yahoo.csv')
    #         ),
    #         Dataset(
    #             description='texas',
    #             file_path=os.path.join(self.DATA_PATH, 'texas.csv')
    #         ),
    #         Dataset(
    #             description='windermere',
    #             file_path=os.path.join(self.DATA_PATH, 'windermere.csv')
    #         ),
    #         Dataset(
    #             description='nky',
    #             file_path=os.path.join(self.DATA_PATH, 'nky.csv')
    #         )
    #     ]
    #
    #     labels = self._get_labels(datasets)
    #
    #     evaluation = ModelEvaluation(model, datasets, labels)
    #     results = evaluation.cross_columns()
    #
    #     print(results)
    #
    #     self.assertEqual(1, 1)


