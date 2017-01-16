try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene

from serene.matcher import SchemaMatcher
import os.path

EXAMPLE_DATASET = os.path.join('tests', 'resources', 'medium.csv')

# connect to the server...
# host and port are for now taken from the config.settings.py file
dm = SchemaMatcher(
    host="localhost",
    port=8080)

#
# lists all the datasets...
#
print(dm.datasets)
# [
#  <DataSet(gr7y7ydf)>,
#  <DataSet(sdfg764t)>,
#  <DataSet(98793875)>
# ]
#
# summary of all datasets as a Pandas data frame
#
print(dm.datasets.summary)
#   dataset_id        name   description     created    modified     num_rows  num_cols
#  4  42fv87g2    name.csv   test1          2016-04-07  2016-04-07       100        20
#  5  8ut20yet   phone.csv   test2          2016-04-07  2016-04-07       100        10
#  6  d55ifgh0  phone2.csv   test3          2016-04-07  2016-04-07       100        10

#
# upload a new dataset
#
new_dataset = dm.create_dataset(
    description="testing",
    file_path=EXAMPLE_DATASET,
    type_map={}
)

print("Created a new dataset:")
print(new_dataset)

print()
print("This should appear in the main set")
print(dm.datasets)

print()
print("We can look at the dataset properties")
print("filename:", new_dataset.filename)
print("id:", new_dataset.id)
print("sample: ", new_dataset.sample)
print("summary:", new_dataset.summary)

print()
print("We can also delete it")
dm.remove_dataset(new_dataset)

print()
print(dm.datasets)


# lists all the models...
#
print()
print("Here is a list of all the models on the server:")
print(dm.models)

#
# summary for all models
#
print()
print("Summary in table form")
print(dm.models.summary)
#     model_id  description     created    modified     status  state_created state_modified
#  1  42fv87g2         name  2016-04-07  2016-04-07   COMPLETE     2016-04-07     2016-04-07
#  2  8ut20yet        phone  2016-04-07  2016-04-07  UNTRAINED     2016-04-07     2016-04-07
#  3  d55ifgh0        phone  2016-04-07  2016-04-07       BUSY     2016-04-07     2016-04-07
#  4  d55ffgh0        phone  2016-04-07  2016-04-07      ERROR     2016-04-07     2016-04-07

print()
print("Summary of a single model")
print(dm.models[0].summary)

#
# build a new model
#
print()
print("Build a new model with parameters...")

#
# features for the classifier
#
features = {
    "activeFeatures" : [
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
}

#
# resampling strategy
#
resampling_strategy = "ResampleToMean"

#
# list of semantic types
#
classes = ["year_built", "address", "bathrooms", "bedrooms", "email", "fireplace", "firm_name", "garage",
           "heating", "house_description", "levels", "mls", "phone", "price", "size", "type", "unknown"]

#
# description of the model
#
descr = "test model"

#
# add some training data...
#
print("We add some training data...")
datasets = [
    dm.create_dataset(
        description="testing 1",
        file_path=EXAMPLE_DATASET,
        type_map={}
    ),
    dm.create_dataset(
        description="testing 2",
        file_path=EXAMPLE_DATASET,
        type_map={}
    ),
]


for d in datasets:
    print(d.filename)

#
# User labelled data
#
label_data = {
    datasets[0].column('Quality'): 'bedrooms',
    datasets[0].column('Year'): 'year_built',
    datasets[0].column('Product code'): 'mls',
    datasets[1].column('Quality'): 'bedrooms',
    datasets[1].column('Year'): 'year_built',
    datasets[1].column('Product code'): 'mls',
}

#
# Create a new model.
#
model = dm.create_model(
    feature_config=features,
    classes=classes,
    description=descr,
    labels=label_data,
    resampling_strategy=resampling_strategy
)
print(model.summary)

print()
print("Now we should see the new model in the list")
print(dm.models)

#
# remove a model...
#
print()
print("We can also remove models")
dm.remove_model(model.id)
print(dm.models)



#
# Create a new model for testing.
#
print("Let's create a model for testing...")
new_model = dm.create_model(
    feature_config=features,
    classes=classes,
    description=descr,
    labels=label_data,
    resampling_strategy=resampling_strategy
)
print(model.summary)


# Training needs to be called explicitly afterwards
#
# show model settings
print()
print(new_model)
#{
#   'classes': ['name', 'address', 'phone', 'unknown'],
#   'cost_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#   'dateCreated': datetime.datetime(2016, 12, 5, 3, 34, 17),
#   'dateModified': datetime.datetime(2016, 12, 13, 3, 6, 54),
#   'description': 'This is the description',
#   'features': {   'activeFeatureGroups': ['prop-instances-per-class-in-knearestneighbours'],
#                   'activeFeatures': ['ratio-alpha-chars'],
#                   'featureExtractorParams': [   {   'name': 'prop-instances-per-class-in-knearestneighbours',
#                                                     'num-neighbours': '3'}]},
#   'id': 1756569785,
#   'labelData': {   '1457824964': 'phone',
#                    '1648830195': 'address',
#                    '1963212042': 'address',
#                    '2139943057': 'phone',
#                    '466311398': 'phone'},
#   'modelPath': '/tmp/storage/models/1756569785/workspace/1756569785.rf',
#   'modelType': 'randomForest',
#   'refDataSets': [1094049002],
#    'resamplingStrategy': 'ResampleToMean',
#   'state': ModelState(Status.COMPLETE, modified on 2016-12-13 03:07:00)
# }

print("by default, model shows the current labels for the dataset columns (ground truth)")

print(new_model.labels)
#       user_label column_name    column_id dataset_id
#  2       unknown      colnma     08y08yfg   23452345
#  4          name        junk     zs6egdia   34575675
#  5         phone          ph     hs6egdib   23452345
#  6         phone  home-phone     xs6egdic   23452345
#  7          name        name     bs6egdid   23452345
#  8       address        addr     gs6egdie   34575675
#  9       address   busn-addr     as6egdif   34575675
# 10          name   junk-name     q25srwty   34575675

#
# add labels
#
print()
print("labels can be added, but re-training has to be called later on")
print("refer to the labels with a column ID or a column object...")
print(new_model.add_labels({
    datasets[0][3]: 'address',
    datasets[0][4]: 'phone'
}))

#
# add labels by column name...
#
print()
print("labels can also be added one by one")
print(new_model.add_label(
    datasets[0].column("Product code"), "phone"
))
#       user_label column_name    column_id dataset_id
#  2       unknown      colnma     08y08yfg  987938475
#  4          name        junk     zs6egdia  987938475
#  5         phone          ph     hs6egdib  216186312
#  6         phone  home-phone     xs6egdic  987938475
#  7          name        name     bs6egdid  987938475
#  8       address        addr     gs6egdie  216186312
#  9       address   busn-addr     as6egdif  216186312
# 10          name   junk-name     q25srwty  216186312
# 11          name    column_1        12342  987938475
# 11          addr    column_2        12343  216186312


# train/re-train model and perform inference
# all datasets will be used
print()
print("Next we can train a model")
print("The initial state for {} is {}".format(new_model.id, new_model.state))
print("Training...")
new_model.train()
print("Done.")
print("The final state for {} is {}".format(new_model.id, new_model.state))

#
# now we can predict for each dataset
print()
print("First we can add some datasets to test:")
evals = [
    dm.create_dataset(
        description="evaluation 1",
        file_path=EXAMPLE_DATASET,
        type_map={}
    ),
    dm.create_dataset(
        description="evaluation 2",
        file_path=EXAMPLE_DATASET,
        type_map={}
    ),
    dm.create_dataset(
        description="evaluation 3",
        file_path=EXAMPLE_DATASET,
        type_map={}
    ),
    dm.create_dataset(
        description="evaluation 4",
        file_path=EXAMPLE_DATASET,
        type_map={}
    )
]

ds = evals[0]
print()
print("We now predict for each dataset")
print(new_model.predict(ds))

print()
print("... or we could use the dataset id")
print(new_model.predict(ds.id))
#     dataset_id    predicted_label  confidence column_name model_id  user_label column_id
# 0   1717853384              price    0.250000      colnma   345683       price     745894
# 1   1717853384          firm_name    0.150000        firm   345683   firm_name     473863
# 2   1717853384               type    0.250000        smth   345683        type     74e894
# 3   1717853384            unknown    0.350000         hel   345683         NaN     84w894
# 4   1717853384            address    0.571429        addr   345683     address     74j894
# 5   1717853384          firm_name    0.600000      agency   345683   firm_name     545564
# 6   1717853384            unknown    0.750000         trd   345683         NaN     445234
# 7   1717853384              email    0.600000       email   345683       email     545844
# 8   2747853384            unknown    0.250000        uurr   345683         NaN     145894

#
# shortcut to view the user labels...
#
print()
print("check the original labels")
print(new_model.labels)
#     model_id   user_label column_name    column_id
#  2  6ff9sdug      unknown      colnma     08y08yfg
#  4  42fv87g2         name        junk     js6egdia
#  5  8ut20yet        phone          ph     js6egdia
#  6  d55ifgh0        phone  home-phone     js6egdia
#  7  vbcuw238         name        name     js6egdia
#  8  jf84hsod      address        addr     js6egdia
#  9  7db29sdj      address   busn-addr     js6egdia
# 10  d0h1hskj         name   junk-name     q25srwty

#
# now the model scores can also be viewed...
#
print()
print("Model scores")
print(new_model.predict(ds, scores=True))
#     dataset_id    predicted_label  confidence column_name model_id  user_label column_id  price firm_name address
# 0   1717853384              price    0.250000      colnma   345683       price     745894  0.25      0.08     0.1
# 1   1717853384          firm_name    0.150000        firm   345683   firm_name     473863  0.05      0.15     0.1
# 2   1717853384               type    0.250000        smth   345683        type     74e894  0.20      0.08     0.1
# 3   1717853384            unknown    0.350000         hel   345683         NaN     84w894  0.25      0.08     0.1
# 4   1717853384            address    0.571429        addr   345683     address     74j894  0.25      0.08   0.571
# 5   1717853384          firm_name    0.600000      agency   345683   firm_name     545564  0.25      0.60     0.1
# 6   1717853384            unknown    0.750000         trd   345683         NaN     445234  0.25      0.08     0.1

#
# the full feature list can also be viewed...
#
print()
print("Model features")
print(new_model.predict(ds, scores=False, features=True))
#           id    label    col_name    dataset  dataset_id     feature_0 alpha_ratio  number_at_signs  number_slashes  digit_ratio  number_of_words  number_of_ats
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1         0.5             0.62             0.0          0.4             0.11            0.0
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1         0.5             0.62             0.0          0.4             0.11            0.0
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0         0.5             0.62             0.0          0.4             0.11            0.0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1         0.5             0.62             0.0          0.4             0.11            0.0
#  4  42fv87g2     name        junk   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
#  7  vbcuw238     name        name   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0         0.5             0.62             0.0          0.4             0.11            0.0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0         0.5             0.62             0.0          0.4             0.11            0.0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0         0.5             0.62             0.0          0.4             0.11            0.0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0         0.5             0.62             0.0          0.4             0.11            0.0
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1         0.5             0.62             0.0          0.4             0.11            0.0
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0         0.5             0.62             0.0          0.4             0.11            0.0

#
# all data shown...
#
print()
print("All model prediction data")
print(new_model.predict(ds, scores=True, features=True))
#           id user_label  predicted_label  col_name    dataset  dataset_id  user_labeled score_addr score_phone score_unknown score_addr
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1       0.06        0.11          0.01       0.83       0.04
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1       0.86        0.12          0.05       0.01       0.04
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0       0.16        0.11          0.16       0.03       0.04
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1       0.02        0.11          0.01       0.83       0.04
#  4  42fv87g2     name        junk   test.csv    js6egdia             0       0.86        0.12          0.01       0.03       0.04
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0       0.45        0.41          0.02       0.01       0.04
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0       0.86        0.17          0.01       0.03       0.04
#  7  vbcuw238     name        name   test.csv    js6egdia             0       0.76        0.14          0.05       0.03       0.04
#  8  jf84hsod  address        addr   test.csv    js6egdia             0       0.36        0.11          0.01       0.01       0.04
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0       0.87        0.16          0.01       0.03       0.04
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0       0.86        0.11          0.02       0.03       0.04
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0       0.26        0.12          0.01       0.01       0.04
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0       0.86        0.11          0.01       0.03       0.04
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1       0.56        0.14          0.02       0.01       0.04
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0       0.82        0.11          0.01       0.03       0.04

#
# compute confusion matrix on a prediction
#
print()
print("Confusion matrix of the predicted values...")
print(dm.confusion_matrix(new_model.predict(ds)))

#
# train all models
#
print()
print("We can also train all models (may take a while...)")
dm.train_all()
print("Done training all models")

#
# get models in the Error state
#
print()
print("Print the models in the error state")
for m in dm.models:
    if m.is_error:
        print("ERROR in model:")
        print(m)

#
# we can also predict across all datasets
#
print()
print("We can also predict across all datasets (may take a while...)")
output = dm.predict_all(new_model)
print(output)
print("Done predicting across everything")

#
# Print the confusion matrix for all output
#
print()
print("Confusion matrix")
print(dm.confusion_matrix(output))
