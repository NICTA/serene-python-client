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
from serene import Column, DataNode, ClassNode, Status

STOPFORINPUT = True
DELETE_DATA = True

def delete_serene_data(sn):

    octopii = sn.octopii.items
    for o in octopii:
        print('Removing octopus: {}'.format(o.id))
        sn.octopii.remove(o.id)

    ssds = sn.ssds.items
    for s in ssds:
        print('Removing ssd: {}'.format(s.id))
        sn.ssds.remove(s.id)

    ontologies = sn.ontologies.items
    for ont in ontologies:
        print('Removing ontology: {}'.format(ont.id))

    models = sn.models.items
    for m in models:
        print('Removing model: {}'.format(m.id))

    datasets = sn.datasets.items
    for d in datasets:
        print('Removing dataset: {}'.format(d.id))

# =======================
#
#  Step 1: Start with a connection to the server...
#
# =======================
sn = serene.Serene(
    host='127.0.0.1',
    port=8080,
)
print(sn)

# Clean out serene
if DELETE_DATA:
    print('Removing old data from serene...')
    delete_serene_data(sn)

# =======================
#
#  Step 2: Upload some training datasets...
#
# =======================
#
# assigning the DataSet to a variable will help us keep track...
#

print()
print('First we upload some datasets...')
data_path = os.path.join(os.path.dirname(__file__), "resources", "data")
owl_path = os.path.join(os.path.dirname(__file__), "resources", "owl")

train_1 = sn.datasets.upload(os.path.join(data_path, 'train1.csv'))
train_2 = sn.datasets.upload(os.path.join(data_path, 'train2.csv'))

datasets = [train_1, train_2]

print('Done.')
#
# lets have a look...
#
for d in datasets:
    print("DataSet:", d.id)
    print("        ", d.description)
    print("        ", d.columns)
    print()

# =======================
#
#  Step 3: Upload some ontologies
#
# =======================

print()
print("Now we handle the ontologies...")
#
# first let's see the ontologies that exist
#
ontologies = sn.ontologies.items

if len(ontologies):
    print()
    print("Displaying first ontology on the server...")
    ontologies[0].show()

ontology_local = serene.Ontology(os.path.join(owl_path, 'bio1_export.ttl'))

ontology_local.show()

# ontology_local.to_turtle(os.path.join(owl_path, 'bio1_export.ttl'))

print()
print('Displaying local ontology...')
if STOPFORINPUT:
    input('Press enter to continue...')

#
# If ok, we can upload this to the server as well...
#
ontology = sn.ontologies.upload(ontology_local)

# ===============
#
#  Step 4: Now start labelling some datasets with SSDs
#
# ===============

# First lets have a look at the datasets...
for d in datasets:
    print("DataSet:", d.id)
    print("         ", d.description)
    print(d.sample)
    print()

# And at the datanodes we'll need to map
print("DataNodes:")
print(ontology.data_nodes)

train_1_ssd = (sn
               .SSD(train_1, ontology, name='train-one')
               .map(Column('first_name'), DataNode(ClassNode('Person'), 'identifier', 0))
               .map(Column('last_name'), DataNode(ClassNode('Person'), 'identifier', 1))
               .map(Column('email'), DataNode(ClassNode('Email'), 'identifier'))
               .map(Column('gender'), DataNode(ClassNode('Person'), 'gender'))
               .map(Column('dob'), DataNode(ClassNode('Person'), 'birthDate'))
               .map(Column('phone'), DataNode(ClassNode('Phone'), 'identifier'))
               .map(Column('address'), DataNode(ClassNode('Address'), 'identifier'))
               .link('Person', 'associatedWith', 'Address')
               .link('Person', 'associatedWith', 'Phone')
               .link('Person', 'associatedWith', 'Email'))

train_1_ssd.show()
print()
print('Displaying train data 1 Semantic Source Description (SSD')
if STOPFORINPUT:
    input('Press enter to continue...')

train_2_ssd = (sn
               .SSD(train_2, ontology, name='train-two')
               .map(Column('fullname'), DataNode(ClassNode('Person'), 'identifier'))
               .map(Column('email'), DataNode(ClassNode('Email'), 'identifier'))
               .map(Column('gender'), DataNode(ClassNode('Person'), 'gender'))
               .map(Column('dob'), DataNode(ClassNode('Person'), 'birthDate'))
               .map(Column('phone'), DataNode(ClassNode('Phone'), 'identifier'))
               .map(Column('address'), DataNode(ClassNode('Address'), 'identifier'))
               .link('Person', 'associatedWith', 'Address')
               .link('Person', 'associatedWith', 'Phone')
               .link('Person', 'associatedWith', 'Email'))

train_2_ssd.show()
print()
print('Displaying train data 1 Semantic Source Description (SSD')
if STOPFORINPUT:
    input('Press enter to continue...')

# upload all these to the server. Here we ignore the return value (the SSD object will be updated)
ssds = [train_1_ssd, train_2_ssd]
for ssd in ssds:
    sn.ssds.upload(ssd)

# ==========
#
#  Step 5. Create an Octopus data integration object...
#
# ==========
# resampling_strategy = ['NoResampling', 'ResampleToMean', 'BaggingToMax', 'Bagging']

octo_local = sn.Octopus(
    ssds=ssds,
    ontologies=[ontology],
    name='octopus-test',
    description='Testing example for typical bio data',
    resampling_strategy="Bagging",  # optional
    num_bags=100,  # optional
    bag_size=10,  # optional
    model_type="randomForest",
    modeling_props={
        'compatibleProperties': True,
        'ontologyAlignment': True,
        'addOntologyPaths': True, # we allow the semantic modeller to use ontology paths
        'mappingBranchFactor': 100,
        'numCandidateMappings': 10,
        'topkSteinerTrees': 10,
        'multipleSameProperty': True,
        'confidenceWeight': 1.0,
        'coherenceWeight': 1.0,
        'sizeWeight': 0.5,
        'numSemanticTypes': 10,
        'thingNode': False,
        'nodeClosure': True,
        'propertiesDirect': True,
        'propertiesIndirect': True,
        'propertiesSubclass': True,
        'propertiesWithOnlyDomain': True,
        'propertiesWithOnlyRange': True,
        'propertiesWithoutDomainRange': True,
        'unnknownThreshold': 0.5
    },
    feature_config={
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
                "max-comparisons-per-class": 3
            }, {
                "name": "min-wordnet-jcn-distance-from-class-examples",
                "max-comparisons-per-class": 3
            }, {
                "name": "min-wordnet-lin-distance-from-class-examples",
                "max-comparisons-per-class": 3
            }
        ]
    }
)

# add this to the endpoint...
print("Now we upload to the server")
octo = sn.octopii.upload(octo_local)

# =======================
#
# Step 6. Train
#
# =======================

print()
print("Next we can train the Octopus")
print("The initial state for {} is {}".format(octo.id, octo.state))
print("Training...")
octo.train()
print("Done.")
print("The final state for {} is {}".format(octo.id, octo.state))

if octo.state.status in {Status.ERROR}:
    print("Something went wrong. Failed to train the Octopus.")
    exit()
