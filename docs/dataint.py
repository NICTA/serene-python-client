import dataint

# connect to the server...
# host and port are for now taken from the config.settings.py file
dm = dataint.SchemaMatcher(
    host="http://localhost",
    port=9000)

# lists all the datasets...
dm.dataset
# {
#   987394875: <MatcherDataSet(gr7y7ydf)>
#   583047856: <MatcherDataSet(sdfg764t)>
#   098098345: <MatcherDataSet(98793875)>
# }

# summary of all datasets as a Pandas data frame
dm.dataset_summary
#   dataset_id        name   description     created    modified     num_rows  num_cols
#  4  42fv87g2    name.csv   test1          2016-04-07  2016-04-07       100        20
#  5  8ut20yet   phone.csv   test2          2016-04-07  2016-04-07       100        10
#  6  d55ifgh0  phone2.csv   test3          2016-04-07  2016-04-07       100        10


# reference the dataset by the id...

dm.dataset[987298345]
# name: asdf.csv
# dateCreated:  2015-02-01 12:30.123
# dateModified: 2015-02-04 16:31.382
# columns: colna, junk, ph, home-phone
# sample:
#
#   colna   junk          ph   home-phone
#    1234    345  0294811111  76382687234
#   98745     as  0287364857  87598374958
#  987394    sjk  0358798753  09183098235
#      22  837fh  0928059834  98734985798
#   34875   kdjf  0980394805  02398098345


# alternatively reference by filename...
# this does not work now

dm.dataset['asdf.csv']
# filename: asdf.csv
# date_created:  2015-02-01 12:30.123
# date_modified: 2015-02-04 16:31.382
# columns: colna, junk, ph, home-phone
# sample:
#
#   colna   junk          ph   home-phone
#    1234    345  0294811111  76382687234
#   98745     as  0287364857  87598374958
#  987394    sjk  0358798753  09183098235
#      22  837fh  0928059834  98734985798
#   34875   kdjf  0980394805  02398098345


# view a column...

dm.dataset['asdf.csv'].sample['junk']
#    1234
#   98745
#  987394
#      22
#   34875
#    1234
#   98745
#  987394
#      22
#   34875
#    1234
#     745
#       4
#      22
#   34875


# upload a new dataset
new_dataset = dm.upload_dataset(description="testing", file_path=filepath, type_map={})

# create a dictionary of label data for the new dataset
# based on the .csv file which has mandatory columns 'column_name' and 'class'
label_data = new_dataset.construct_labelData('some_file_with_column_labels.csv')


# lists all the models...

dm.model
# {
#    1234: Model(1234),
#    2345: Model(2345),
#    3456: Model(3456),
#    567: Model(567),
#    4678: Model(4678)
# }

# summary for all models

dm.model_summary
#     model_id  description     created    modified     status  state_created state_modified
#  1  42fv87g2         name  2016-04-07  2016-04-07   COMPLETE     2016-04-07     2016-04-07
#  2  8ut20yet        phone  2016-04-07  2016-04-07  UNTRAINED     2016-04-07     2016-04-07
#  3  d55ifgh0        phone  2016-04-07  2016-04-07       BUSY     2016-04-07     2016-04-07
#  4  d55ffgh0        phone  2016-04-07  2016-04-07      ERROR     2016-04-07     2016-04-07



# Configuration parameters for the model upload

# configuration for the feature calculation
features = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals"],
            "activeFeatureGroups": ["stats-of-text-length", "prop-instances-per-class-in-knearestneighbours"],
            "featureExtractorParams": [
                {"name": "prop-instances-per-class-in-knearestneighbours", "num-neighbours": 5}]
            }

# resampling strategy
resamplingStrategy = "ResampleToMean"

# list of semantic types
classes = ["year_built", "address", "bathrooms", "bedrooms", "email", "fireplace", "firm_name", "garage",
           "heating", "house_description", "levels", "mls", "phone", "price", "size", "type", "unknown"]

#description of the model
descr = "test model"

# Create a new model.

model = dm.create_model(
    feature_config=features, classes=classes, description=descr,
    labels=label_data, resampling_strategy=resamplingStrategy
) # Training needs to be called explicitly afterwards

# show model settings
print(model)
# model_key: 345683
# model_type: randomForest
# features_config: {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals"],
#                   "activeFeatureGroups": ["stats-of-text-length", "prop-instances-per-class-in-knearestneighbours"],
#                   "featureExtractorParams": [
#                   {"name": "prop-instances-per-class-in-knearestneighbours", "num-neighbours": 5}]}
# cost_matrix: []
# resampling_strategy: ResampleToMean
# label_data: {}
# ref_datasets: {1726,17636}
# classes: ['unknown', 'address', 'bathrooms', 'bedrooms']
# date_created: 2016-08-11 03:02:52
# date_modified: 2016-08-11 03:02:52
# model_state: ModelState(<Status.COMPLETE: 'complete'>, created on 2016-08-11 03:02:52, modified on 2016-08-11 03:02:53, message '')



# by default model shows the current labels for the dataset columns (ground truth)

model
#           id  actual_label column_name    column_id
#  2  6ff9sdug      unknown      colnma     08y08yfg
#  4  42fv87g2         name        junk     js6egdia
#  5  8ut20yet        phone          ph     js6egdia
#  6  d55ifgh0        phone  home-phone     js6egdia
#  7  vbcuw238         name        name     js6egdia
#  8  jf84hsod      address        addr     js6egdia
#  9  7db29sdj      address   busn-addr     js6egdia
# 10  d0h1hskj         name   junk-name     q25srwty


# labels can be added, but re-training has to be called later on

model.add_labels({'12342': 'name', '12343': 'addr'},dm.session)

#           id  actual_label column_name    column_id
#  2  6ff9sdug      unknown      colnma     08y08yfg
#  4  42fv87g2         name        junk     js6egdia
#  5  8ut20yet        phone          ph     js6egdia
#  6  d55ifgh0        phone  home-phone     js6egdia
#  7  vbcuw238         name        name     js6egdia
#  8  jf84hsod      address        addr     js6egdia
#  9  7db29sdj      address   busn-addr     js6egdia
# 10  d0h1hskj         name   junk-name     q25srwty

# train/re-train model and perform inference
model.get_predictions(dm.session)


# Now the model contains the predictions...

model
#     dataset_id    predicted_label  confidence column_name model_id actual_label column_id
# 0   1717853384              price    0.250000      colnma   345683       price     745894
# 1   1717853384          firm_name    0.150000        firm   345683   firm_name     473863
# 2   1717853384               type    0.250000        smth   345683        type     74e894
# 3   1717853384            unknown    0.350000         hel   345683         NaN     84w894
# 4   1717853384            address    0.571429        addr   345683     address     74j894
# 5   1717853384          firm_name    0.600000      agency   345683   firm_name     545564
# 6   1717853384            unknown    0.750000         trd   345683         NaN     445234
# 7   1717853384              email    0.600000       email   345683       email     545844
# 8   2747853384            unknown    0.250000        uurr   345683         NaN     145894


# shortcut to view the user labels...

model.user_labels # currently
model.get_labeldata()
#     model_id  actual_label column_name    column_id
#  2  6ff9sdug      unknown      colnma     08y08yfg
#  4  42fv87g2         name        junk     js6egdia
#  5  8ut20yet        phone          ph     js6egdia
#  6  d55ifgh0        phone  home-phone     js6egdia
#  7  vbcuw238         name        name     js6egdia
#  8  jf84hsod      address        addr     js6egdia
#  9  7db29sdj      address   busn-addr     js6egdia
# 10  d0h1hskj         name   junk-name     q25srwty


# the model scores can also be viewed...

model.scores
#     dataset_id    predicted_label  confidence column_name model_id actual_label column_id  price firm_name address email  unknown type ...
# 0   1717853384              price    0.250000      colnma   345683       price     745894  0.25       0.08     0.1  0.03      0.1  0.1
# 1   1717853384          firm_name    0.150000        firm   345683   firm_name     473863  0.05       0.15     0.1  0.03      0.1  0.1
# 2   1717853384               type    0.250000        smth   345683        type     74e894  0.20       0.08     0.1  0.03      0.1 0.25
# 3   1717853384            unknown    0.350000         hel   345683         NaN     84w894  0.25       0.08     0.1  0.03     0.35  0.1
# 4   1717853384            address    0.571429        addr   345683     address     74j894  0.25       0.08   0.571  0.03      0.1  0.1
# 5   1717853384          firm_name    0.600000      agency   345683   firm_name     545564  0.25       0.60     0.1  0.03      0.1  0.1
# 6   1717853384            unknown    0.750000         trd   345683         NaN     445234  0.25       0.08     0.1  0.03     0.75  0.1


# the full feature list can also be viewed...

model.features
#           id    label    col_name    dataset  dataset_id  user_labeled alpha_ratio  number_at_signs  number_slashes  digit_ratio  number_of_words  number_of_ats
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


# all data shown...

model.all_data
#           id    actual_label  predicted_label  col_name    dataset  dataset_id  user_labeled score_addr score_phone score_unknown score_addr score_name features...
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


# compute confusion matrix
# possible if training has been done
confusion_matrix = model.calculate_confusionMatrix()

# train all models
dm.train_models()
# True # if all models have been successfully trained

# perform prediction with all models
# training is done automatically
dm.get_predictions_models()

# get models in the Error state
dm.error_models()
# {
#    1234: Model(1234),
#    2345: Model(2345),
#    3456: Model(3456),
#    567: Model(567),
#    4678: Model(4678)
# }




###################################################################################################

#
# OLD STUFF
#




# shows the current labels...
dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1

# show the histories...
# dm.commits
# 98y98wy  + - branch: master
#          |
# 19y3uoa  +
#          |
# h7dhd92  +
#          |\
# bcus79z  | + - branch: test
#          | |
# nhdgsfd  + |
#          | |
# dfs7fti  | +
#          |
# jdhf8eh  o

# list all the datasets
dm.datasets
# [Dataset<08y08yfg>]

# show details of a single dataset
dm.dataset("08y08yfg")
# Dataset 08y08yfg
# Created: 2006-02-14
# Format: CSV
# Rows: 2387685
# Columns 4
# Cols:
#   kjhsdfs
#       wer
#    colnma
#      junk

# list the model configs...
dm.models
# >> [
#     (Model: 98ty9fgudfsy
#      features:
#        name
#        addr
#        unknown
#        addr
#      file: model1.rf
#      labels: a, b, c, d)
#
#     (Model: 98ty9fgudfsy
#      features:
#       name
#       addr
#       unknown
#       addr
#     file: model1.rf
#     labels: a, b, c, d)
#   ]
#

# current model..
dm.model
# >> Model: 98ty9fgudfsy
# >>  features:
# >>    name
# >>    addr
# >>  file: model1.rf
# >>  labels: a, b, c, d


# show the histories...
# dm.commits
# 98y98wy  + - branch: master
#          |
# 19y3uoa  +
#          |
# h7dhd92  +
#          |\
# bcus79z  | + - branch: test
#          | |
# nhdgsfd  + |
#          | |
# dfs7fti  | +
#          |
# jdhf8eh  o

# grab a point in the history
dm.checkout(id="h7dhd92")
dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2  unknown     kjhsdfs  names.csv    08y08yfg             0
#  1  djdifgh0  unknown         wer  names.csv    08y08yfg             0
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod  unknown        junk  names.csv    08y08yfg             0

# nothing was labeled, lets go back to the head of master
dm.checkout(id=None, branch="master")
dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1

# show the current accuracy...
dm.accuracy
# >> 0.95153

# show the current model...
dm.model
# >> <Model 97ty1293r8fy>
#    features - something, something seomthing
#    target - something

# add a new dataset...
dm.add_dataset("test.csv")
# >> <DIFrame id="9f0fdg89h98uhr3g" branch="master">
#    <DataSet id="123y773597y2t1bc" name="test.csv">
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet  unknown          ph   test.csv    js6egdia             0
#  6  d55ifgh0  unknown  home-phone   test.csv    js6egdia             0
#  7  vbcuw238  unknown        name   test.csv    js6egdia             0
#  8  jf84hsod  unknown        addr   test.csv    js6egdia             0
#  9  7db29sdj     name   busn-addr   test.csv    js6egdia             0

# the commits will update with the new dataset...
dm.commits()
# 98y98wy  + --- branch: master
#          |
# 19y3uoa  +
#          |
# h7dhd92  +
#          |\
# bcus79z  | + - branch: test
#          | |
# nhdgsfd  + |
#          | |
# dfs7fti  | +
#          | |
# jdhf8eh  + |
#          | |
# kd82gsq  | + - test.csv added
#          |
# 8fhwbsk  o   - test.csv added

# filter on the junk column names...
dm[dm['name'] == "junk"]
#           id    label    col_name    dataset  dataset_id  user_labeled
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2  unknown        junk   test.csv    js6egdia             0

# label a single column
dm.label(column="junk", dataset="test.csv", label="name")
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     name     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             1
#  5  8ut20yet  unknown          ph   test.csv    js6egdia             0
#  6  d55ifgh0  unknown  home-phone   test.csv    js6egdia             0
#  7  vbcuw238  unknown        name   test.csv    js6egdia             0
#  8  jf84hsod  unknown        addr   test.csv    js6egdia             0
#  9  7db29sdj     name   busn-addr   test.csv    js6egdia             0

# label it by the id...
dm.label(id="asdf87g2", label="addr")
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet  unknown          ph   test.csv    js6egdia             0
#  6  d55ifgh0  unknown  home-phone   test.csv    js6egdia             0
#  7  vbcuw238  unknown        name   test.csv    js6egdia             0
#  8  jf84hsod  unknown        addr   test.csv    js6egdia             0
#  9  7db29sdj     name   busn-addr   test.csv    js6egdia             0

# label them in bulk...
dm.label([("test.csv", "ph", "phone"),
          ("test.csv", "home-phone", "phone"),
          ("test.csv", "name", "name"),
          ("test.csv", "addr", "address"),
          ("test.csv", "business addr", "address")])
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0

dm.commits()
# 98y98wy  + --- branch: master
#          |
# 19y3uoa  +
#          |
# h7dhd92  +
#          |\
# bcus79z  | + - branch: test
#          | |
# nhdgsfd  + |
#          | |
# dfs7fti  | +
#          | |
# jdhf8eh  + |
#          | |
# kd82gsq  + | - test.csv added
#          | |
# 8fhwbsk  | + - test.csv added
#          |
# w7q19swn +   - labels added
#          |
# gjlzh095 +   - labels added
#          |
# kxz26s9b o   - labels added


# now see the junk changes...
dm[dm['name'] == "junk"]
#           id    label    col_name    dataset  dataset_id  user_labeled
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0

# add a new dataset....
dm.add_dataset("file://junk.csv")
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0

dm.checkout()
dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0

for idx in dm['id']:
    dm.add_label(idx, 'blank-label')
dm
#           id        label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2  blank-label     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0  blank-label         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  blank-label      colnma  names.csv    08y08yfg             0
#  3  jf837sod  blank-label        junk  names.csv    08y08yfg             1
#  4  42fv87g2  blank-label        junk   test.csv    js6egdia             0
#  5  8ut20yet  blank-label          ph   test.csv    js6egdia             0
#  6  d55ifgh0  blank-label  home-phone   test.csv    js6egdia             0
#  7  vbcuw238  blank-label        name   test.csv    js6egdia             0
#  8  jf84hsod  blank-label        addr   test.csv    js6egdia             0
#  9  7db29sdj  blank-label   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj  blank-label   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  blank-label       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  blank-label       12334   junk.csv    q25srwty             0
# 13  1ejhs0sd  blank-label   junk-addr   junk.csv    q25srwty             1
# 14  a7dybs14  blank-label     another   junk.csv    q25srwty             0

# remains the same
dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0

dm.features
#           id    label    col_name    dataset  dataset_id  user_labeled alpha_ratio  number_at_signs  number_slashes  digit_ratio  number_of_words  number_of_ats
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


dm.scores
#           id    label    col_name    dataset  dataset_id  user_labeled score_addr score_phone score_unknown score_addr score_name
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


dm.all
#           id    label    col_name    dataset  dataset_id  user_labeled score_addr score_phone score_unknown score_addr score_name
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

dm
#           id    label    col_name    dataset  dataset_id  user_labeled
#  0  asdf87g2     addr     kjhsdfs  names.csv    08y08yfg             1
#  1  djdifgh0    phone         wer  names.csv    08y08yfg             1
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  3  jf837sod     addr        junk  names.csv    08y08yfg             1
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 13  1ejhs0sd     addr   junk-addr   junk.csv    q25srwty             1
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0

# dm setting model parameters
dm.set_model(name="new-model",
             features=['name', 'full-name'])

dm.commits()
# 98y98wy    + --- branch: master
#            |
# 19y3uoa    +
#            |
# h7dhd92    +
#            |\
# bcus79z    | + - branch: test
#            | |
# nhdgsfd    + |
#            | |
# dfs7fti    | +
#            | |
# jdhf8eh    + |
#            | |
# kd82gsq    | + - test.csv added
#            |
# 8fhwbsk    +
#           /|
# gzfwosq  | +   - branch: master
#          |
# kd82gsq  o     - branch: new-model


# types...
# di.Model
# di.DataSet
# di.LabelData
# di.DIFrame


