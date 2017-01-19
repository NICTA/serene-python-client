"""
This module provides model evaluation constructs.

..module:: eval

Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>
"""

from collections import namedtuple
from tempfile import mkstemp
from os import remove
from os.path import basename, join
from glob import iglob
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score


DatasetSpec = namedtuple('DatasetSpec', 'name path')
ColumnLabelSpec = namedtuple('ColumnLabelSpec', 'dataset column label')


def average_accuracy(y_true, y_pred):
    """
    Compute the average accuracy.

    .. function:: average_accuracy(y_true, y_pred)
       :param y_true: The true labels.
       :type y_true: list(str)
       :param y_pred: The predicted labels.
       :type y_pred: list(str)
       :return: The average accuracy.
       :rtype: float
    """
    labels = set(y_true) | set(y_pred)
    pairs = list(zip(y_true, y_pred))
    scores = []

    for label in labels:
        true_positive = sum(
            1 for pair in pairs if pair == (label, label)
        )

        true_negative = sum(
            1 for pair in pairs if pair[0] != label and pair[1] != label
        )

        false_positive = sum(
            1 for pair in pairs if pair[0] != label and pair[1] == label
        )

        false_negative = sum(
            1 for pair in pairs if pair[0] == label and pair[1] != label
        )

        s = true_positive + true_negative + false_positive + false_negative
        scores.append((true_positive + true_negative) / s)

    return sum(scores) / len(scores)


def error_rate(y_true, y_pred):
    """
    Compute the error rate.

    .. function:: error_rate(y_true, y_pred)
       :param y_true: The true labels.
       :type y_true: list(str)
       :param y_pred: The predicted labels.
       :type y_pred: list(str)
       :return: The error rate.
       :rtype: float
    """
    labels = set(y_true) | set(y_pred)
    pairs = list(zip(y_true, y_pred))
    scores = []

    for label in labels:
        true_positive = sum(
            1 for pair in pairs if pair == (label, label)
        )

        true_negative = sum(
            1 for pair in pairs if pair[0] != label and pair[1] != label
        )

        false_positive = sum(
            1 for pair in pairs if pair[0] != label and pair[1] == label
        )

        false_negative = sum(
            1 for pair in pairs if pair[0] == label and pair[1] != label
        )

        s = true_positive + true_negative + false_positive + false_negative
        scores.append((false_positive + false_negative) / s)

    return sum(scores) / len(scores)


class ModelEvaluation:
    """
    The class provides model evaluation utilities.

    .. class:: ModelEvaluation(model, datasets, column_label_map)
       :param model: The model to be evaluated.
       :type model: Model
       :param dataset_specs: The specs of the datasets used for evaluation.
       :type dataset_specs: list(DatasetSpec)
       :param column_label_specs: The specs of the columns and true labels.
       :type column_label_specs: list(ColumnLabelSpec)
       :param schema_matcher: A SchemaMatcher instance used for talking to API.
       :type schema_matcher: SchemaMatcher
    """

    def __init__(self,
                 model,
                 dataset_specs,
                 column_label_specs,
                 schema_matcher):
        """Initialize a class instance."""
        self._model = model
        self._dataset_specs = dataset_specs
        self._column_label_specs = column_label_specs
        self._schema_matcher = schema_matcher

    def cross_columns(self, k=10):
        """
        Do a K-Fold cross-validation based on all columns.

        .. method:: cross_columns(k=10)
           :param k: The number of folds to use for cross-validation.
           :type k: int
           :return: The scores resulting from evaluating the model.
           :rtype: dict(str, float)
        """
        self._datasets = {
            spec.name: read_csv(spec.path)
            for spec in self._dataset_specs
        }

        self._columns = DataFrame(
            {'dataset': dataset_name, 'column': column_name}
            for dataset_name, data_frame in self._datasets.items()
            for column_name in data_frame.columns.tolist()
        )

        self._column_labels = DataFrame(self._column_label_specs)

        results = []

        for training_indices, test_indices in self._kfold(k, self._columns):
            training_columns = self._columns.loc[training_indices]
            test_columns = self._columns.loc[test_indices]

            datasets = self._create_datasets(
                self._extract_columns(training_columns),
                self._extract_columns(test_columns)
            )
            training_dataset, test_dataset = datasets

            model = self._create_model(training_columns, training_dataset)

            model.train()

            prediction_labels = self._extract_prediction_labels(
                model.predict(test_dataset), test_dataset, test_columns
            )

            test_labels = self._extract_test_labels(test_columns)

            results.append(
                self._evaluate_result(test_labels, prediction_labels)
            )

            self._clean_up(model, datasets)

        return self._average_result(results)

    def _kfold(self, k, columns):
        return KFold(n_splits=k).split(columns)

    def _extract_columns(self, columns):
        data_frame_extracts = []

        for dataset_name, data_frame in self._datasets.items():
            columns_to_extract = columns[columns['dataset'] == dataset_name]
            columns_to_extract = columns_to_extract['column'].tolist()

            if len(columns_to_extract) == 0:
                continue

            data_frame_extract = data_frame[columns_to_extract]

            data_frame_extract.columns = [
                self._full_column_name(column_name, dataset_name)
                for column_name in data_frame_extract.columns.tolist()
            ]

            data_frame_extracts.append(data_frame_extract)

        return concat(data_frame_extracts, axis=1)

    def _full_column_name(self, column_name, dataset_name):
        return '%s@%s' % (column_name, dataset_name)

    def _create_datasets(self, training_data, test_data):
        _, training_temp = mkstemp(suffix='.csv')
        _, test_temp = mkstemp(suffix='.csv')

        training_data.to_csv(training_temp, index=False)
        test_data.to_csv(test_temp, index=False)

        training_dataset = self._schema_matcher.create_dataset(
            description='MODEL_EVALUATION_TRAINING',
            file_path=training_temp,
            type_map={}
        )

        test_dataset = self._schema_matcher.create_dataset(
            description='MODEL_EVALUATION_TEST',
            file_path=test_temp,
            type_map={}
        )

        remove(training_temp)
        remove(test_temp)

        return (training_dataset, test_dataset)

    def _create_model(self, training_columns, training_dataset):
        def get_column(dataset_name, column_name):
            return training_dataset.column(
                self._full_column_name(column_name, dataset_name)
            )

        join_index = ['dataset', 'column']

        training_labels = training_columns.join(
            self._column_labels.set_index(join_index),
            on=join_index,
            how='inner'
        )

        classes = list(set(training_labels['label'].tolist()) | {'unknown'})

        labels = {
            get_column(record['dataset'], record['column']): record['label']
            for record in training_labels.to_dict(orient='records')
        }

        return self._schema_matcher.create_model(
            description=self._model.description + '#EVALUATION',
            feature_config=self._model.features,
            resampling_strategy=self._model.resampling_strategy,
            classes=classes,
            labels=labels
        )

    def _extract_prediction_labels(
            self, prediction, test_dataset, test_columns):
        prediction = prediction[['column_id', 'label']].to_records(index=False)

        column_id_name_map = {
            column.id: column.name for column in test_dataset.columns
        }

        prediction = {
            column_id_name_map[column_id]: label
            for column_id, label in prediction
        }

        return {
            column_name:
            prediction.get(column_name, 'unknown')
            for column_name in (
                self._full_column_name(record['column'], record['dataset'])
                for record in test_columns.to_dict(orient='records')
            )
        }

    def _extract_test_labels(self, test_columns):
        def normalize_label(label):
            if type(label) is float:
                return 'unknown'
            return label

        join_index = ['dataset', 'column']

        test_labels = test_columns.join(
            self._column_labels.set_index(join_index),
            on=join_index,
            how='left'
        )

        return {
            self._full_column_name(record['column'], record['dataset']):
            normalize_label(record['label'])
            for record in test_labels.to_dict(orient='records')
        }

    def _evaluate_result(self, test_labels, prediction_labels):
        def get_sorted_labels(column_label_map):
            return [
                item[1] for item in
                sorted(column_label_map.items(), key=lambda item: item[0])
            ]

        y_true = get_sorted_labels(test_labels)
        y_pred = get_sorted_labels(prediction_labels)

        return {
            'average_accuracy':
                average_accuracy(y_true, y_pred),
            'error_rage':
                error_rate(y_true, y_pred),
            'precision_micro':
                precision_score(y_true, y_pred, average='micro'),
            'precision_macro':
                precision_score(y_true, y_pred, average='macro'),
            'recall_micro':
                recall_score(y_true, y_pred, average='micro'),
            'recall_macro':
                recall_score(y_true, y_pred, average='macro'),
            'f1_micro':
                f1_score(y_true, y_pred, average='micro'),
            'f1_macro':
                f1_score(y_true, y_pred, average='macro')
        }

    def _clean_up(self, model, datasets):
        self._schema_matcher.remove_model(model.id)
        for dataset in datasets:
            self._schema_matcher.remove_dataset(dataset)

    def _average_result(self, results):
        return DataFrame(results).mean().to_dict()


def load_specs_from_dir(dir, labels_file_name='labels.csv'):
    """
    Load dataset and label specs from a data directory.

    .. function:: load_specs_from_dir(dir, labels_file_name='labels.csv')
       :param dir: The path to the directory containing all dataset CSV files
                   and the column-label spec CSV file.
       :type dir: str
       :param labels_file_name: The name of the column-label spec CSV file.
       :type labels_file_name: str
       :return: The loaded dataset specs and column-label specs.
       :rtype: tuple
    """
    dataset_spec_file_paths = (
        file_path
        for file_path in iglob(join(dir, '*.csv'))
        if basename(file_path) != labels_file_name
    )

    dataset_specs = [
        DatasetSpec(name=basename(file_path), path=file_path)
        for file_path in dataset_spec_file_paths
    ]

    column_label_specs = [
        ColumnLabelSpec(**record) for record in
        read_csv(join(dir, labels_file_name)).to_dict(orient='records')
    ]

    return (dataset_specs, column_label_specs)
