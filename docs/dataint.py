
import dataint

# connect to the server...
dm = dataint.DataMatcher(
    host="http://localhost",
    port=9000)

# lists all the datasets...
dm.dataset
# {
#   987394875: <DataSet(gr7y7ydf)>
#   583047856: <DataSet(sdfg764t)>
#   098098345: <DataSet(98793875)>
# }


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

dm.dataset['asdf.csv']
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


# lists all the models...

dm.models
# [
#    Model(1234),
#    Model(2345),
#    Model(3456),
#    Model(567),
#    Model(4678)
# ]


# Create a new model. Trains up a model behind the scene

model = di.create_model(
    name="New Model",
    type=ModelType.RANDOM_FOREST,
    col_types=['name', 'addr', 'phone']
)

# show model settings
model.info
# id
# modelType
# classes
# features
# costMatrix
# resamplingStrategy
# labelData
# refDaraSets
# state
# dateCreated
# dateModified


# by default model shows the current labels for the datasets
# in case there are no labels/predictions it will be empty?

model
#           id    label    col_name    dataset  dataset_id  user_labeled
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0


# labels can be added, which will re-train a model

model.add_labels({'colnma': 'name', 'junk': 'addr'})

#           id    label    col_name    dataset  dataset_id  user_labeled
#  2  6ff9sdug     name      colnma  names.csv    08y08yfg             1
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
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0


# Now the model contains the predictions...

model
#           id    label    col_name    dataset  dataset_id  user_labeled
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg             0
#  4  42fv87g2     name        junk   test.csv    js6egdia             0
#  5  8ut20yet    phone          ph   test.csv    js6egdia             0
#  6  d55ifgh0    phone  home-phone   test.csv    js6egdia             0
#  7  vbcuw238     name        name   test.csv    js6egdia             0
#  8  jf84hsod  address        addr   test.csv    js6egdia             0
#  9  7db29sdj  address   busn-addr   test.csv    js6egdia             0
# 10  d0h1hskj     name   junk-name   junk.csv    q25srwty             0
# 11  91ytgfd0  unknown       dflgk   junk.csv    q25srwty             0
# 12  zf84hsod  unknown       12334   junk.csv    q25srwty             0
# 14  a7dybs14  unknown     another   junk.csv    q25srwty             0


# add explicitly predictions???

model.predictions

# the accuracy can be obtained...

model.accuracy
# 0.87


# shortcut to view the user labels...

model.user_labels
#           id    label    col_name    dataset  dataset_id
#  2  6ff9sdug  unknown      colnma  names.csv    08y08yfg
#  4  42fv87g2     name        junk   test.csv    js6egdia


# the model scores can also be viewed...

model.scores
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

model.all
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


