"""
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

from serene import Ontology, DataNode, Mapping, Link, Column, ClassNode, Transform

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

# assigning the DataSet to a variable will help us keep track...
#
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "data")
owl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "resources", "owl")

business_info = sn.datasets.upload(os.path.join(data_path, 'businessInfo.csv'))
employee_address = sn.datasets.upload(os.path.join(data_path, 'EmployeeAddresses.csv'))
get_cities = sn.datasets.upload(os.path.join(data_path, 'getCities.csv'))
get_employees = sn.datasets.upload(os.path.join(data_path, 'getEmployees.csv'))
postal_code = sn.datasets.upload(os.path.join(data_path, 'postalCodeLookup.csv'))

datasets = [business_info, employee_address, get_cities, get_employees, postal_code]

input("Press enter to continue")
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

# =======================
#
#  Step 3: Upload some ontologies
#
# =======================

#
# first let's see the ontologies that exist
#
ontologies = sn.ontologies.items

if len(ontologies):
    ontologies[0].show()

    input("Press enter to continue")

# #
# # The most straightforward way is to upload an OWL file directly...
# #
# ontology = sn.ontologies.upload('tests/resources/owl/dataintegration_report_ontology.ttl')
#
# ontology.show()
#
# input("Press enter to continue...")
#
# #
# # Or if you want to check it first you can do...
# #
# local_ontology = serene.Ontology('tests/resources/owl/dataintegration_report_ontology.ttl')
#
# local_ontology.show()
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
ontology_local = (
    serene.Ontology()
          .uri("http://www.semanticweb.org/serene/report_example_ontology")
          .class_node("Place", ["name","postalCode"])
          .class_node("City", is_a="Place")
          .class_node("Person", {"name": str, "phone": int, "birthDate": datetime.datetime})
          .class_node("Organization", {"name": str, "phone": int, "email": str})
          .class_node("Event", {"startDate": datetime.datetime, "endDate": datetime.datetime})
          .class_node("State", is_a="Place")
          .link("Person", "bornIn", "Place")
          .link("Organization", "ceo", "Person")
          .link("Place", "isPartOf", "Place")
          .link("Person", "livesIn", "Place")
          .link("Event", "location", "Place")
          .link("Organization", "location", "Place")
          .link("Organization", "operatesIn", "City")
          .link("Place", "nearby", "Place")
          .link("Event", "organizer", "Person")
          .link("City", "state", "State")
          .link("Person", "worksFor", "Organization"))
#
# Have a quick look...
#
ontology_local.show()

#
# or output to an owl file if you want...
ontology_local.to_turtle(os.path.join(owl_path, 'test.ttl'))


#
input("Press enter to continue...")

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

business_info_ssd = (sn.SSD(business_info, ontology)
                     .map(Column("company"), DataNode("Organization", "name"))
                     .map(Column("ceo"), DataNode("Person", "name"))
                     .map(Column("city"), DataNode("City", "name"))
                     .map(Column("state"), DataNode("State", "name"))
                     .link("Organization", "City", relationship="operatesIn")
                     .link("Organization", "Person", relationship="ceo")
                     .link("City", "State", relationship="state"))

business_info_ssd.show()

input("press any key to continue...")

#
# We also have just a string shorthand...
#
employee_address_ssd = (sn.SSD(employee_address, ontology)
                        .map("name", "Person.name")
                        .map("address", "Place.name")
                        .map("postcode", "Place.postalCode")
                        .link("Person", "Place", relationship="livesIn"))

employee_address_ssd.show()

input("press any key to continue...")

get_cities_ssd = (sn.SSD(get_cities, ontology)
                  .map(Column("city"), DataNode("City", "name"))
                  .map(Column("state"), DataNode("State", "name"))
                  .link("City", "State", relationship="state"))

get_cities_ssd.show()

input("press any key to continue...")

get_employees_ssd = (sn.SSD(get_employees, ontology)
                     .map(Column("employer"), DataNode("Organization", "name"))
                     .map(Column("employee"), DataNode("Person", "name"))
                     .link("Person", "Organization", relationship="worksFor"))


get_employees_ssd.show()

input("press any key to continue...")


postal_code_ssd = (sn.SSD(postal_code, ontology)
                   .map(Column("zipcode"), DataNode("Place", "postalCode"))
                   .map(Column("city"), DataNode("City", "name"))
                   .map(Column("state"), DataNode("State", "name"))
                   .link("City", "State", relationship="state")
                   .link("City", "Place", relationship="isPartOf"))

postal_code_ssd.show()

input("press any key to continue...")

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
    name='octopus-test',
    description='Testing example for places and companies',
    ontologies=[ontology],
    resampling_strategy="NoResampling",  # optional
    num_bags=100,  # optional
    bag_size=10,  # optional
    model_type="randomForest",
    modeling_props={

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

# =======================
#
#  Step 7. Run predictions
#
# =======================

personal_info = sn.datasets.upload('tests/resources/data/personalInfo.csv')
predicted = octo.predict(personal_info, candidates=3)

for p in predicted:
    print("Predicted candidate rank", p.score.rank)
    print("Score:")
    p.score.show()
    p.ssd.show()
    input("Press any key to continue...")

# the best is number 0!
predicted_ssd = predicted[0].ssd

# =======================
#
#  Step 7. Correct the predicted result...
#
# =======================

predicted_ssd.remove(Mapping(Column("name"), DataNode("Place", "name")))
predicted_ssd = predicted_ssd.map(Column("name"), DataNode("Person", "name"))

predicted_ssd.show()

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

print(octo.mappings)
print()
print(octo.mappings[DataNode("Person", "name")])
print()

person_columns = [col for node, col in octo.mappings.items() if node == DataNode("Person", "name")]

print(person_columns)
print()





















#
# print()
# print("Ontology built:")
# print(on)
# on.show()
#
# print()
# print("output to .owl file")
# on.to_turtle()
#
# input("Press enter to continue...")
#
# # add the ontology to the SemanticModeller
# sm.add_ontology(on)
#
# # list all the class nodes
# print()
# print("Listing class nodes...")
# for cn in sm.class_nodes:
#     print(cn)
#
# # list all the data nodes
# print()
# print("Listing data nodes...")
# for dn in sm.data_nodes:
#     print(dn)
#
# # list all the links
# print()
# print("Listing the links...")
# for link in sm.links:
#     print(link)
#
# # ==========================
# #  CREATING SEMANTIC MODELS
# # ==========================
#
# # once we load the file, the columns exist in the SemanticSourceDescription(SSD),
# # but they have no DataNodes assigned...
# ssd = sm.to_ssd("tests/resources/data/EmployeeAddresses.csv")
#
# print()
# print("Printing ssd...")
# print(ssd)
#
# # show the original file location...
# print()
# print("Printing filename...")
# print(ssd.file)
#
# # show the data (a Pandas data frame)
# #print("Printing dataframe...")
# #print(ssd.df)
#
# # show the initial mapping
# print()
# print("Printing mapping...")
# for m in ssd.mappings:
#     print(m)
#
#
# # you can assign the DataNodes to columns manually...
# print()
# print("Adding one map...")
# ssd.map(Column("postcode"), DataNode("postcode"))
# print(ssd)
#
# # you can assign the DataNodes with a transform...
# print()
# print("Ambiguities in the DataNode name will lead to an error:")
# try:
#     ssd.map(Column("FirstName"), DataNode("name"))
# except LookupError as err:
#     print("Error: {0}".format(err))
# print()
# print("This can be resolved by specifying the class node ")
#
# ssd.map(Column("FirstName"), DataNode("Person", "name"))
#
# print(ssd)
#
# # you can assign the DataNodes with a transform...
# print()
# print("We can also add transforms")
# ssd.map(Column("FirstName"), DataNode("Person", "name"), transform=lambda x: x.lower())
# ssd.map(Column("address"), DataNode("Address", "name"), transform=lambda x: x.upper())
# print(ssd)
#
# # transforms
# print()
# print("Printing single transform:")
# print(ssd.transforms[0])
#
# print()
# print("Printing single transform with find:")
# print(ssd.find(Transform(6)))
#
# print()
# print("Printing all transforms:")
# print(ssd.transforms)
#
# # links
#
# # we can also add links
# print()
# print("We can link ClassNodes manually:")
# ssd.link(ClassNode("Person"), ClassNode("Address"), relationship="lives-at")
# ssd.link(ClassNode("Address"), ClassNode("Person"), relationship="owned-by")
# print(ssd)
#
# print()
# print("Though note that we can only choose links from the ontology:")
# try:
#     ssd.link(ClassNode("Person"), ClassNode("Address"), relationship="non-existing-link")
# except Exception as err:
#     print("Error: {0}".format(err))
# print()
#
# # samples...
# print()
# print("You can also sample columns")
# print(ssd.sample(Column("address")))
# print()
# print("You can add transforms to samples as a preview")
# print(ssd.sample(Column("address"), Transform(6)))
#
# # we can also try to auto-fill the SSD
# ssd.predict()
# print()
# print("Predicted model...")
# print(ssd)
#
# # we can just show the predicted models too
# print()
# print("Show just the predictions...")
# for p in ssd.predictions:
#     print(p)
#
# # we can then correct it
# print()
# print("Now we can correct the model manually if necessary")
# ssd.map(Column("LastName"), DataNode("Person", "last-name"))
# print(ssd)
#
# # we can also remove the elements
# print()
# print("We can remove links with lookups...")
# ssd.remove(Link("owned-by"))
# print(ssd)
#
# # list all the data nodes
# print()
# print("Listing data nodes...")
# for dn in ssd.data_nodes:
#     print(dn)
#
#
# # we can also remove the elements
# # print()
# # print("We can remove nodes with DataNode lookups...")
# # ssd.remove(DataNode("Person", "last-name"))
# # print(ssd)
#
# # list all the data nodes
# print()
# print("Listing data nodes...")
# for dn in ssd.data_nodes:
#     print(dn)
#
# # display
# print()
# print("We can also display the semantic model...")
# ssd.show()
#
# # save
# print()
# print("We can also save the semantic source description...")
# ssd.save(file="out.ssd")
#
# # we can access the model:
# print()
# print("The semantic model")
# print(ssd.model)
#
# print()
# print("The semantic model links")
# print(ssd.model.links)
#
# print()
# print("The semantic model nodes")
# print(ssd.model.class_nodes)
#
# # Or print out the json
# print()
# print("The source description JSON:")
# print(ssd.json)
#
# # Or print out the json
# print()
# print("The source description dictionary:")
# print(ssd.ssd['semanticModel'])
#
# print()
# print("Columns")
# print(ssd.columns)
#
# print()
# print("Mappings")
# print(ssd.mappings)
