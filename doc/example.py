"""

attr_id,class
cooling@residential_listing_-_feature_details@house_listing@homeseekers.csv,cooling
cooling@house_listing@nky.csv,cooling
ac@house_listing@windermere.csv,cooling
house_location@house_listing@nky.csv,address
location@residential_listing_-_basic_features@house_listing@homeseekers.csv,address
address@house_listing@windermere.csv,address
house_location@house_listing@yahoo.csv,address
house_location@house_listing@texas.csv,address

company@file.csv,Organization.name
ceo@file.csv,Person.name
file.csv,Organization-ceo-Person
file.csv,Organization-location-City
file.csv,Person-livesIn-City


business_info_ssd = (sn
                     .SSD(business_info, ontology, name='business-info')
                     .map(Column("company"), DataNode(ClassNode("Organization"), "name"))
                     .map(Column("ceo"), DataNode(ClassNode("Person"), "name"))
                     .map(Column("city"), DataNode(ClassNode("City"), "name"))
                     .map(Column("state"), DataNode(ClassNode("State"), "name"))
                     .link("Organization", "operatesIn", "City")
                     .link("Organization", "ceo", "Person")
                     .link("City", "state", "State"))

business_info_ssd.show()

print()
print("Displaying businessInfo.csv Semantic Source Description (SSD)")
input("press any key to continue...")

#
# We also have just a string shorthand...
#

ontology_file = 'tests/resources/owl/dataintegration_report_ontology.ttl'

datasets = [
    'tests/resources/data/businessInfo.csv',
    'tests/resources/data/EmployeeAddresses.csv',
    'tests/resources/data/getCities.csv',
    'tests/resources/data/getEmployees.csv',
    'tests/resources/data/postalCodeLookup.csv',
]

map_file = tests/resources/data/example-map-file.csv

> cat example-map-file.csv

attr_id, class
name@EmployeeAdddress.csv, Person.name
address@EmployeeAdddress.csv, Place.name
postcode@EmployeeAdddress.csv, Place.postalCode
getCities.csv@city, City.name
getCities.csv@state, State.name
employer@getEmployees.csv, Organization.name
employee@getEmployees.csv, Person.name
zipcode@postalCodeLookup.csv, Place.postalCode
city@postalCodeLookup.csv, City.name
state@postalCodeLookup.csv, State.name

link_file = tests/resources/data/example-link-file.csv

> cat example-link-file.csv

file, src, type, dst
EmployeeAdddress.csv, Person, livesIn, Place
getCities.csv, City, state, State
getEmployees.csv, Person, worksFor, Organization
postalCodeLookup.csv, City, state, State
postalCodeLookup.csv, City, isPartOf, Place


 input
  Some knowledge about the ontology or where you want to end up...
  /some/junk/test1.csv  - contains columns: name, birth, city, state, workplace
  /some/junk/test2.csv  - contains columns: state, city
  /some/junk/test3.csv  - contains columns: company, ceo, city, state
  /some/junk/test4.csv  - contains columns: employer, employee
  /some/junk/test5.csv  - contains columns: zipcode, city, state
  ..
  ..
  ..
  ...
 output
  SSD: semantic source description files
    /some/junk/test1.ssd
    /some/junk/test2.ssd
    /some/junk/test3.ssd
    /some/junk/test4.ssd
    /some/junk/test5.ssd
    ...
    ...
    ...
"""

# ===========
#    SETUP
# ===========
try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene

import datetime
import pandas as pd
import os

from serene import SSD, Status, DataProperty, Mapping, ObjectProperty, Column, Class, DataNode, ClassNode

# We have these datasets:
#
# businessInfo.csv company,ceo,city,state
# EmployeeAddresses.csv: FirstName,LastName,address,postcode
# getCities: state,city
# getEmployees: employer,employee
# postalCodeLookup: zipcode,city,state
#
# and we want to predict these...
#
# personalInfo: name,birthDate,city,state,workplace
#


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
#
# >>> Serene(127.0.0.1:9000)
#

# =======================
#
#  Step 2: Upload some training datasets...
#
# =======================
#
# assigning the DataSet to a variable will help us keep track...
#
print()
print("First we upload some datasets...")
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "data")
owl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "owl")

data_path = os.path.join(os.curdir, "tests", "resources", "data")
owl_path = os.path.join(os.curdir, "tests", "resources", "owl")


business_info = sn.datasets.upload(os.path.join(data_path, 'businessInfo.csv'))
employee_address = sn.datasets.upload(os.path.join(data_path, 'EmployeeAddresses.csv'))
get_cities = sn.datasets.upload(os.path.join(data_path, 'getCities.csv'))
get_employees = sn.datasets.upload(os.path.join(data_path, 'getEmployees.csv'))
postal_code = sn.datasets.upload(os.path.join(data_path, 'postalCodeLookup.csv'))

datasets = [business_info, employee_address, get_cities, get_employees, postal_code]

print("Done.")
#
# lets have a look...
#
for d in datasets:
    print("DataSet:", d.id)
    print("        ", d.description)
    print("        ", d.columns)
    print()

#
# employee_address is not so great, we need to combine first and last name. We
# can use pandas for this
#
print()
print("Here we edit a dataset and re-upload...")

sn.datasets.remove(employee_address)
df = pd.read_csv(os.path.join(data_path, 'EmployeeAddresses.csv'))
# move to first name/last name
df['name'] = df[['FirstName', 'LastName']].apply(
    lambda row: row[0] + ' ' + row[1], axis=1
)
del df['FirstName']
del df['LastName']
#
# now we can re-upload
#
employee_address = sn.datasets.upload(df)

print("Done")

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

    # input("Press enter to continue")

# #
# # The most straightforward way is to upload an OWL file directly...
# #
# ontology = sn.ontologies.upload('tests/resources/owl/dataintegration_report_ontology.ttl')
#
# print("Showing the uploaded ontology...")
#
# ontology.show()
#
# input("Press enter to continue...")
#
# #
# # Or if you want to check it first you can do...
# #
#local_ontology = serene.Ontology('tests/resources/owl/dataintegration_report_ontology.ttl')
ontology_local = serene.Ontology(os.path.join(owl_path, 'paper.ttl'))

#
# local_ontology.show()
#
# print("Showing the local ontology...")
#
# input("Press enter to continue...")
#
# #
# # Now we can upload
# #
# ontology2 = sn.ontologies.upload(local_ontology)

#
# Or you can build one programmatically
#
# ontology_local = (
#     serene.Ontology()
#           .uri("http://www.semanticweb.org/serene/example_ontology")
#           .owl_class("Place", ["name", "postalCode"])
#           .owl_class("City", is_a="Place")
#           .owl_class("Person", {"name": str, "phone": int, "birthDate": datetime.datetime})
#           .owl_class("Organization", {"name": str, "phone": int, "email": str})
#           .owl_class("Event", {"startDate": datetime.datetime, "endDate": datetime.datetime})
#           .owl_class("State", is_a="Place")
#           .link("Person", "bornIn", "Place")
#           .link("Organization", "ceo", "Person")
#           .link("Place", "isPartOf", "Place")
#           .link("Person", "livesIn", "Place")
#           .link("Event", "location", "Place")
#           .link("Organization", "location", "Place")
#           .link("Organization", "operatesIn", "City")
#           .link("Place", "nearby", "Place")
#           .link("Event", "organizer", "Person")
#           .link("City", "state", "State")
#           .link("Person", "worksFor", "Organization"))

# Have a quick look...
#
ontology_local.show()
input("press any key to continue...")
#
# or output to an owl file if you want...
ontology_local.to_turtle(os.path.join(owl_path, 'test.ttl'))


#
print()
print("Displaying local ontology...")
# input("Press enter to continue...")

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


# businessInfo.csv company,ceo,city,state
# EmployeeAddresses.csv: name,address,postcode
# getCities: state,city
# getEmployees: employer,employee
# postalCodeLookup: zipcode,city,state

business_info_ssd = (sn
                     .SSD(business_info, ontology, name='business-info')
                     .map(Column("company"), DataNode(ClassNode("Organization"), "name"))
                     .map(Column("ceo"), DataNode(ClassNode("Person"), "name"))
                     .map(Column("city"), DataNode(ClassNode("City"), "name"))
                     .map(Column("state"), DataNode(ClassNode("State"), "name"))
                     .link("Organization", "operatesIn", "City")
                     .link("Organization", "ceo", "Person")
                     .link("City", "state", "State"))

business_info_ssd.show(outfile="businessInfo.dot")

print()
print("Displaying businessInfo.csv Semantic Source Description (SSD)")
# input("press any key to continue...")

#
# We also have just a string shorthand...
#
employee_address_ssd = (sn
                        .SSD(employee_address, ontology, name='employee-addr')
                        .map("name", "Person.name")
                        .map("address", "Place.name")
                        .map("postcode", "Place.postalCode")
                        .link("Person", "livesIn", "Place"))

employee_address_ssd.show()

print()
print("Displaying EmployeeAdddress.csv Semantic Source Description (SSD)")
# input("press any key to continue...")

get_cities_ssd = (sn
                  .SSD(get_cities, ontology, name='cities')
                  .map(Column("city"), DataNode(ClassNode("City"), "name"))
                  .map(Column("state"), DataNode(ClassNode("State"), "name"))
                  .link("City", "state", "State"))

get_cities_ssd.show()

print()
print("Displaying getCities.csv Semantic Source Description (SSD)")
# input("press any key to continue...")

get_employees_ssd = (sn
                     .SSD(get_employees, ontology, name='employees')
                     .map(Column("employer"), DataNode(ClassNode("Organization"), "name"))
                     .map(Column("employee"), DataNode(ClassNode("Person"), "name"))
                     .link("Person", "worksFor", "Organization"))


get_employees_ssd.show()

print()
print("Displaying getEmployees.csv Semantic Source Description (SSD)")
# input("press any key to continue...")

postal_code_ssd = (sn
                   .SSD(postal_code, ontology, name='postal-code')
                   .map(Column("zipcode"), "Place.postalCode")
                   .map(Column("city"), "City.name")
                   .map(Column("state"), "State.name")
                   .link("City", "state", "State")
                   .link("City", "isPartOf", "Place")
                   )

postal_code_ssd.show()

print()
print("Displaying postalCodeLookup.csv Semantic Source Description (SSD)")
# input("press any key to continue...")

postal_code_ssd2 = (sn
                    .SSD(postal_code, ontology, name='postal-code')
                    .map(Column("zipcode"), "Place.postalCode")
                    .map(Column("city"), "City.name")
                    .map(Column("state"), "State.name")
                    .link("City", "state", "State")
                    .link("City", "isPartOf", "Place")
                    )

postal_code_ssd2.show()


# upload all these to the server. Here we ignore the return value (the SSD object will be updated)

for ssd in [business_info_ssd, employee_address_ssd, get_cities_ssd, get_employees_ssd, postal_code_ssd]:
    sn.ssds.upload(ssd)


# ==========
#
#  Step 5. Create an Octopus data integration object...
#
# ==========

octo_local = sn.Octopus(
    ssds=[
        business_info_ssd,
        employee_address_ssd,
        get_cities_ssd,
        get_employees_ssd,
        postal_code_ssd
    ],
    ontologies=[ontology],
    name='octopus-test',
    description='Testing example for places and companies',
    resampling_strategy="NoResampling",  # optional
    num_bags=100,  # optional
    bag_size=10,  # optional
    model_type="randomForest",
    modeling_props={
        "compatibleProperties": True,
        "ontologyAlignment": True,
        "addOntologyPaths": True,
        "mappingBranchingFactor": 50,
        "numCandidateMappings": 10,
        "topkSteinerTrees": 50,
        "multipleSameProperty": True,
        "confidenceWeight": 1.0,
        "coherenceWeight": 1.0,
        "sizeWeight": 0.5,
        "numSemanticTypes": 10,
        "thingNode": False,
        "nodeClosure": True,
        "propertiesDirect": True,
        "propertiesIndirect": True,
        "propertiesSubclass": True,
        "propertiesWithOnlyDomain": True,
        "propertiesWithOnlyRange": True,
        "propertiesWithoutDomainRange": False,
        "unknownThreshold": 0.05
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
            "entropy-for-discrete-values",
            "shannon-entropy"
        ],
        "activeFeatureGroups": [
            "inferred-data-type",
            "char-dist-features",
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

# =======================
#
#  Step 7. Run predictions
#
# =======================

personal_info = sn.datasets.upload(os.path.join(data_path,'Employees.csv'))
predicted = octo.predict(personal_info)

print()
print("Predicted schema matcher results::")
print()
schema_results = octo.matcher_predict(personal_info, features=True)
print(schema_results)
schema_results.to_csv("employees_features.csv", index=False)
input("Press any key to continue...")

print()
print("Predicted results::")
print()
for pred in predicted:
    print(pred.score)
    pred.ssd.show()
    #input("Press enter to continue...")



# =======================
#
#  Step 7. Correct the predicted result...
#
# =======================

print("the best is number 0!")
print("Fixing birthdate column")

predicted_ssd = predicted[0].ssd
predicted_ssd.show()

print("Showing first guess...")
input("Press any key to continue...")

predicted_ssd.remove(Column("birthDate"))
predicted_ssd.show()

print("Removing birth date assignment")
input("Press any key to continue...")

predicted_ssd.map(Column("birthDate"), DataNode(ClassNode("Person"), "birthDate"))
predicted_ssd.link("Person", "worksFor", "Organization")
predicted_ssd.show(outfile="personInfo.dot")

print("Adding correct birthdate node")
input("Press any key to continue...")

# =======================
#
#  Step 8. Confirm the result and add to the training data on the server...
#
# =======================

new_ssd = sn.ssds.upload(predicted_ssd)
octo = sn.octopii.update(octo.add(new_ssd))

# =======================
#
#  Step 9. Extract the mappings
#
# =======================

maps = octo.mappings(DataNode(ClassNode("Person"), "name"))

print()
print("These are the columns mapped to Person.name:")
for dn in maps:
    print(dn, "which is in", dn.dataset)

print()

