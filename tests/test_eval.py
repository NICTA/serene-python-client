"""This is a test script of model evaluation module."""


def _get_labels(data_dir_path, datasets):
    from pandas import read_csv, concat
    from serene.eval import Column

    labels = read_csv(data_dir_path + 'labels.csv')

    column_dataset_series = labels['attr_id']

    dataset_series = column_dataset_series \
        .map(lambda column_dataset: column_dataset.rpartition('@')[2]) \
        .map(lambda dataset: dataset[:-4])

    column_series = column_dataset_series.map(
        lambda column_dataset: column_dataset.rpartition('@')[0]
    )

    labels = concat([dataset_series, column_series, labels['class']], axis=1)
    labels.columns = ['dataset', 'column', 'label']

    labels = labels.to_records(index=False)

    datasets = {dataset.description: dataset for dataset in datasets}

    return {
        Column(name=column, dataset=datasets[dataset]): label
        for (dataset, column, label) in labels
        if dataset in datasets
    }


def _test(data_dir_path):
    from serene.eval import ModelEvaluation, Dataset, Model

    model = Model(
        description='evaluation model',
        features={
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
                    "max-comparisons-per-class": 20
                }, {
                    "name": "min-wordnet-jcn-distance-from-class-examples",
                    "max-comparisons-per-class": 20
                }, {
                    "name": "min-wordnet-lin-distance-from-class-examples",
                    "max-comparisons-per-class": 20
                }
            ]
        },
        resampling_strategy="ResampleToMean"
    )

    datasets = [
        Dataset(
            description='homeseekers',
            file_path=data_dir_path + 'homeseekers.csv'),
        Dataset(
            description='yahoo',
            file_path=data_dir_path + 'yahoo.csv'),
        Dataset(
            description='texas',
            file_path=data_dir_path + 'texas.csv'),
        Dataset(
            description='windermere',
            file_path=data_dir_path + 'windermere.csv'),
        Dataset(
            description='nky',
            file_path=data_dir_path + 'nky.csv')
    ]

    labels = _get_labels(data_dir_path, datasets)

    evaluation = ModelEvaluation(model, datasets, labels)
    results = evaluation.cross_columns()
    print(results)

_test('/Users/lichaoir/Dev/serene/matcher/new-csv-data/')

