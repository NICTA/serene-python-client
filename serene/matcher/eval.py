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
       :rtype: tuple(DatasetSpec, ColumnLabelSpec)
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


def get_sorted_labels(column_label_map):
    """
    Sort labels using their corresponding column names.

    .. function: get_sorted_labels(column_label_map)
       :param column_label_map: The column-name-to-label map.
       :type column_label_map: dict(str, str)
       :return: The sorted labels.
       :rtype: list(str)
    """
    return [
        item[1] for item in
        sorted(column_label_map.items(), key=lambda item: item[0])
    ]


def get_sorted_label_candidates(predictions):
    """
    Extract top k labels and sort them using their corresponding column names.

    .. function:: get_sorted_label_candidates(prediction)
       :param prediction: The prediction data frame given by the Serene API.
       :type prediction: DataFrame
       :return: The n * k matrix of top k labels for each column.
       :rtype: ndarray
    """
    scores_seq = predictions.filter(like='scores_').to_dict(orient='records')
    columns = predictions['column_id'].values

    labels_seq = (
        list(
            label.partition('scores_')[2]
            for label, score in
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ) for scores in scores_seq
    )

    return [
        item[1] for item in
        sorted(zip(columns, labels_seq), key=lambda item: item[0])
    ]


def scores(y_true, y_pred):
    """
    Calculate evaluation scores based on true labels and predicted labels.

    .. function: scores(y_true, y_pred)
       :param y_true: The true labels.
       :type y_true: list(str)
       :param y_pred: The predicted labels.
       :type y_pred: list(str)
       :return: The scores.
       :rtype: dict(str, float)
    """
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


def precision_at_k(y_true, y_pred, *, average='micro', k=3):
    """
    Calculate the precision-at-k metric.

    .. function:: precision_at_k(y_true, y_pred, *, average='micro', k=3)
       :param y_true: The true labels.
       :type y_true: list(str)
       :param y_pred: The n * k matrix of predicted labels.
       :type y_pred: ndarray(str)
       :param average: Either 'micro' or 'macro'.
       :type average: str
       :param k: The k value.
       :type k: int
       :return: The precision-at-k value.
       :rtype: float
    """
    y_pred = [pred_labels[:k] for pred_labels in y_pred]

    if average == 'micro':
        rel_pred = 0
        all_pred = 0
        for true_label, pred_labels in zip(y_true, y_pred):
            for pred_label in pred_labels:
                all_pred += 1
                if pred_label == true_label:
                    rel_pred += 1
                    break
        return rel_pred / all_pred
    else:
        labels = set(y_true)
        partitions = {label: [] for label in labels}
        for true_label, pred_labels in zip(y_true, y_pred):
            partitions[true_label].append((true_label, pred_labels))

        precisions = []
        for partition in partitions.values():
            rel_pred = 0
            all_pred = 0
            for true_label, pred_labels in partition:
                for pred_label in pred_labels:
                    all_pred += 1
                    if pred_label == true_label:
                        rel_pred += 1
                        break
            precisions.append(rel_pred / all_pred)
        return precisions / len(labels)


def mrr(y_true, y_pred, *, average='micro'):
    """
    Calculate the mean reciprocal rank metric.

    .. function:: mrr(y_true, y_pred, *, average='micro')
       :param y_true: The true labels.
       :type y_true: list(str)
       :param y_pred: The n * k matrix of predicted labels.
       :type y_pred: ndarray(str)
       :param average: Either 'micro' or 'macro'.
       :type average: str
       :return: The MRR value.
       :rtype: float
    """
    if average == 'micro':
        sum_of_rank = 0
        for true_label, pred_labels in zip(y_true, y_pred):
            try:
                sum_of_rank += 1 / (pred_labels.index(true_label) + 1)
            except ValueError: pass
        return sum_of_rank / len(y_true)
    else:
        labels = set(y_true)
        partitions = {label: [] for label in labels}
        for true_label, pred_labels in zip(y_true, y_pred):
            partitions[true_label].append((true_label, pred_labels))

        scores = []
        for partition in partitions.values():
            sum_of_rank = 0
            for true_label, pred_labels in partition:
                try:
                    sum_of_rank += 1 / (pred_labels.index(true_label) + 1)
                except ValueError: pass
            scores.append(sum_of_rank / len(partition))

        return average(scores)


class CrossColumnEvaluation:
    """
    The class implements K-Fold cross-validation based on all columns.

    .. class:: CrossColumnEvaluation(model, datasets, column_label_map)
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

    def evaluate(self, k=10):
        """
        Do a cross-validation and give the scores.

        .. method:: evaluate(k=10)
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
            prediction = model.predict(test_dataset, scores=True)

            prediction_labels = get_sorted_labels(
                self._extract_prediction_labels(
                    prediction, test_dataset, test_columns
                )
            )

            test_labels = get_sorted_labels(
                self._extract_test_labels(test_columns)
            )

            result = scores(test_labels, prediction_labels)

            # ranked_labels = get_sorted_label_candidates(prediction)
            # result['precision@k'] = precision_at_k(test_labels, ranked_labels)
            # result['mrr'] = mrr(test_labels, ranked_labels)

            results.append(result)

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

    def _clean_up(self, model, datasets):
        self._schema_matcher.remove_model(model.id)
        for dataset in datasets:
            self._schema_matcher.remove_dataset(dataset)

    def _average_result(self, results):
        return DataFrame(results).mean().to_dict()


class CrossDatasetEvaluation():
    """
    The class implements K-Fold cross-validation based on all datasets.

    The number of folds K is chosen to be equal to the number of datasets.

    .. class:: CrossColumnEvaluation(model, datasets, column_label_map)
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

    def evaluate(self):
        """
        Do a cross-validation and give the scores.

        .. method:: evaluate()
           :return: The scores resulting from evaluating the model.
           :rtype: dict(str, float)
        """
        self._datasets = {
            spec: self._schema_matcher.create_dataset(
                description=spec.name + '#EVALUATION',
                file_path=spec.path,
                type_map={}
            )
            for spec in self._dataset_specs
        }

        results = [
            self._evaluate_fold(spec)
            for spec in self._datasets
        ]

        return DataFrame(results).mean().to_dict()

    def _evaluate_fold(self, test_dataset_spec):
        model = self._create_model([
            spec
            for spec in self._dataset_specs
            if spec is not test_dataset_spec
        ])

        test_dataset = self._datasets[test_dataset_spec]

        model.train()
        prediction = model.predict(test_dataset, scores=True)
        self._schema_matcher.remove_model(model.id)

        predicted_labels = get_sorted_labels(
            self._extract_prediction_labels(prediction, test_dataset)
        )

        test_labels = get_sorted_labels(
            self._extract_test_labels(test_dataset_spec, test_dataset)
        )

        result = scores(test_labels, predicted_labels)

        # ranked_labels = get_sorted_label_candidates(prediction)
        # result['precision@k'] = precision_at_k(test_labels, ranked_labels)
        # result['mrr'] = mrr(test_labels, ranked_labels)

        return result

    def _create_model(self, training_dataset_specs):
        dataset_name_to_spec_map = {
            spec.name: spec for spec in training_dataset_specs
        }

        labels = {
            self._datasets[dataset_spec].column(label_spec.column):
            label_spec.label
            for label_spec, dataset_spec in (
                (label_spec, dataset_name_to_spec_map.get(label_spec.dataset))
                for label_spec in self._column_label_specs
            )
            if dataset_spec is not None
        }

        classes = list(set(labels.values()) | {'unknown'})

        return self._schema_matcher.create_model(
            description=self._model.description + '#EVALUATION',
            feature_config=self._model.features,
            resampling_strategy=self._model.resampling_strategy,
            classes=classes,
            labels=labels
        )

    def _extract_prediction_labels(self, prediction, test_dataset):
        prediction = {
            record['column_id']: record['label'] for record in
            prediction[['column_id', 'label']].to_dict(orient='records')
        }

        return {
            column.name:
            prediction.get(column.id, 'unknown')
            for column in test_dataset.columns
        }

    def _extract_test_labels(self, test_dataset_spec, test_dataset):
        labels = {
            spec.column: spec.label
            for spec in self._column_label_specs
            if spec.dataset == test_dataset_spec.name
        }

        return {
            column.name:
            labels.get(column.name, 'unknown')
            for column in test_dataset.columns
        }
