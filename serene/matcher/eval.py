"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Serene Python client: model evaluation module
"""

from collections import namedtuple

Dataset = namedtuple('Dataset', 'description file_path')
Column = namedtuple('Column', 'name dataset')
Model = namedtuple('Model', 'description features resampling_strategy')


class ModelEvaluation:
    """Evaluates model quality with cross-validation."""

    def __init__(
            self,
            model,
            datasets,
            column_label_map,
            schema_matcher_host='localhost',
            schema_matcher_port=8080):
        """Initialize a ModelEvaluation instance."""
        from .core import SchemaMatcher

        self._model = model
        self._datasets = datasets[:]
        self._column_label_map = column_label_map.copy()
        self._data_frames = None
        self._schema_matcher = SchemaMatcher(
            schema_matcher_host, schema_matcher_port
        )

    def cross_datasets(self):
        """
        Do a Leave-One-Out cross-validation.

        This method treats each dataset as a sample.
        """
        pass

    def cross_columns(self, k=10):
        """
        Do a K-Fold cross-validation.

        This method treats each column as a sample.
        """
        self._data_frames = self._read_data()

        columns = [
            Column(name=column_name, dataset=dataset)
            for dataset, data_frame in zip(self._datasets, self._data_frames)
            for column_name in data_frame.columns.tolist()
        ]

        results = []

        for training, test in self._kfold(k, columns):
            training_columns = [columns[i] for i in training]
            test_columns = [columns[i] for i in test]

            datasets = self._create_datasets(
                self._extract_columns(training_columns),
                self._extract_columns(test_columns)
            )
            training_dataset, test_dataset = datasets

            classes = self._get_classes()
            labels = self._get_labels(training_dataset)
            model = self._create_model(classes, labels)

            model.train()

            prediction = self._extract_prediction(
                model.predict(test_dataset), test_dataset, test_columns
            )

            true_labels = self._extract_true_labels(test_columns)

            results.append(self._evaluate_result(true_labels, prediction))

            self._clean_up(model, datasets)

        for result in results:
            for name in sorted(result):
                print(name, result[name])

        return self._average_result(results)

    def _read_data(self):
        from pandas import read_csv
        return [
            read_csv(dataset.file_path)
            for dataset in self._datasets
        ]

    def _kfold(self, k, columns):
        from sklearn.model_selection import KFold
        return KFold(n_splits=k).split(columns)

        # return (
        #     (
        #         self._extract_columns([columns[i] for i in training]),
        #         self._extract_columns([columns[i] for i in test])
        #     )
        #     for training, test in KFold(n_splits=k).split(columns)
        # )

    def _extract_columns(self, columns):
        from operator import contains
        from functools import partial
        from pandas import concat

        data_frame_extracts = []

        for dataset, data_frame in zip(self._datasets, self._data_frames):
            columns_to_extract = set(
                column.name
                for column in columns if column.dataset == dataset
            )

            should_extract = partial(contains, columns_to_extract)
            data_frame_extract = data_frame.select(should_extract, axis=1)

            data_frame_extract.columns = [
                self._full_column_name(column_name, dataset.description)
                for column_name in data_frame_extract.columns.tolist()
            ]

            data_frame_extracts.append(data_frame_extract)

        return concat(data_frame_extracts, axis=1)

    def _create_datasets(self, training, test):
        from tempfile import mkstemp
        from os import remove

        _, training_temp = mkstemp(suffix='.csv')
        _, test_temp = mkstemp(suffix='.csv')

        training.to_csv(training_temp, index=False)
        test.to_csv(test_temp, index=False)

        training_dataset = self._schema_matcher.create_dataset(
            description='model_evaluation_training',
            file_path=training_temp,
            type_map={}
        )

        test_dataset = self._schema_matcher.create_dataset(
            description='model_evaluation_test',
            file_path=test_temp,
            type_map={}
        )

        remove(training_temp)
        remove(test_temp)

        return (training_dataset, test_dataset)

    def _get_classes(self):
        return list(
            set(self._column_label_map.values()) | {'unknown'}
        )

    def _get_labels(self, dataset):
        labels = {}

        for column, label in self._column_label_map.items():
            try:
                dataset_column = dataset.column(self._full_column_name(
                    column.name, column.dataset.description
                ))
            except ValueError:
                continue

            labels[dataset_column] = label

        return labels

    def _full_column_name(self, column_name, dataset_description):
        return '%s@%s' % (column_name, dataset_description)

    def _create_model(self, classes, labels):
        return self._schema_matcher.create_model(
            description=self._model.description,
            feature_config=self._model.features,
            resampling_strategy=self._model.resampling_strategy,
            classes=classes,
            labels=labels
        )

    def _clean_up(self, model, datasets):
        self._schema_matcher.remove_model(model.id)
        for dataset in datasets:
            self._schema_matcher.remove_dataset(dataset)

    def _evaluate_result(self, true_labels, prediction):
        def get_sorted_labels(column_label_map):
            return [
                item[1] for item in
                sorted(column_label_map.items(), key=lambda item: item[0])
            ]

        true_labels = get_sorted_labels(true_labels)
        prediction = get_sorted_labels(prediction)

        result = {
            'average_accuracy': self._average_accuracy(true_labels, prediction),
            'error_rage': self._error_rate(true_labels, prediction)
        }

        from sklearn.metrics import precision_score
        result['precision_micro'] = precision_score(
            true_labels, prediction, average='micro'
        )
        result['precision_macro'] = precision_score(
            true_labels, prediction, average='macro'
        )

        from sklearn.metrics import recall_score
        result['recall_micro'] = recall_score(
            true_labels, prediction, average='micro'
        )
        result['recall_macro'] = recall_score(
            true_labels, prediction, average='macro'
        )

        from sklearn.metrics import f1_score
        result['f1_micro'] = f1_score(
            true_labels, prediction, average='micro'
        )
        result['f1_macro'] = f1_score(
            true_labels, prediction, average='macro'
        )

        return result

    def _average_accuracy(self, true_labels, prediction):
        labels = set(true_labels) | set(prediction)
        pairs = list(zip(true_labels, prediction))
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

    def _error_rate(self, true_labels, prediction):
        labels = set(true_labels) | set(prediction)
        pairs = list(zip(true_labels, prediction))
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

    def _extract_prediction(self, prediction, dataset, test_columns):
        prediction = prediction \
            .filter(['column_id', 'label']) \
            .to_records(index=False)

        column_id_name_map = \
            {column.id: column.name for column in dataset.columns}

        prediction = {
            column_id_name_map[column_id]: label
            for column_id, label in prediction
        }

        return {
            fullname:
            prediction.get(fullname, 'unknown')
            for fullname in (
                self._full_column_name(column.name, column.dataset.description)
                for column in test_columns
            )
        }

    def _extract_true_labels(self, test_columns):
        return {
            self._full_column_name(column.name, column.dataset.description):
            self._column_label_map.get(column, 'unknown')
            for column in test_columns
        }

    def _average_result(self, results):
        from pandas import DataFrame
        return DataFrame(results).mean().to_dict()
