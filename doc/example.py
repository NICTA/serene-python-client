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
    
from serene import SemanticModeller, Ontology, DataNode, Link, Column, ClassNode, Transform

# start with a SemanticModeller object...
sm = serene.SemanticModeller(
    config={
        "kmeans": 5,
        "threshold": 0.7,
        "search-depth": 100
    },
    ontologies=["/some/file/something.owl", "/some/other/file/something.owl"]
)
print(sm)


# build up a simple ontology
on = (serene.Ontology()
      .prefix("asd", "asd:/9e8rgyusdgfiuhsdf.dfguhudfh")
      .uri("http://www.semanticweb.org/data_integration_project/report_example_ontology")
      .class_node("Person", ["first-name", "last-name", "name", "phone"])
      .class_node("Location", prefix="xsd")
      .class_node("City", ["name", "zipcode"], is_a="Location")
      .class_node("State", ["name"], is_a="Location")
      .class_node("Address", {"name": str, "postcode": int}, is_a="Location")
      .link("Person", "lives-at", "Address")
      .link("Address", "owned-by", "Person")
      .link("Address", "is_in", "City")
      .link("Address", "is_in", "State"))

print()
print("Ontology built:")
print(on)

# add the ontology to the SemanticModeller
sm.add_ontology(on)

# list all the class nodes
print()
print("Listing class nodes...")
for cn in sm.class_nodes:
    print(cn)

# list all the data nodes
print()
print("Listing data nodes...")
for dn in sm.data_nodes:
    print(dn)

# list all the links
print()
print("Listing the links...")
for link in sm.links:
    print(link)

# ==========================
#  CREATING SEMANTIC MODELS
# ==========================

# once we load the file, the columns exist in the SemanticSourceDescription(SSD),
# but they have no DataNodes assigned...
ssd = sm.to_ssd("tests/resources/data/EmployeeAddresses.csv")

print()
print("Printing ssd...")
print(ssd)

# show the original file location...
print()
print("Printing filename...")
print(ssd.file)

# show the data (a Pandas data frame)
#print("Printing dataframe...")
#print(ssd.df)

# show the initial mapping
print()
print("Printing mapping...")
for m in ssd.mappings:
    print(m)


# you can assign the DataNodes to columns manually...
print()
print("Adding one map...")
ssd.map(Column("postcode"), DataNode("postcode"))
print(ssd)

# you can assign the DataNodes with a transform...
print()
print("Ambiguities in the DataNode name will lead to an error:")
try:
    ssd.map(Column("FirstName"), DataNode("name"))
except LookupError as err:
    print("Error: {0}".format(err))
print()
print("This can be resolved by specifying the class node ")

ssd.map(Column("FirstName"), DataNode("Person", "name"))

print(ssd)

# you can assign the DataNodes with a transform...
print()
print("We can also add transforms")
ssd.map(Column("FirstName"), DataNode("Person", "name"), transform=lambda x: x.lower())
ssd.map(Column("address"), DataNode("Address", "name"), transform=lambda x: x.upper())
print(ssd)

# transforms
print()
print("Printing single transform:")
print(ssd.transforms[0])

print()
print("Printing single transform with find:")
print(ssd.find(Transform(6)))

print()
print("Printing all transforms:")
print(ssd.transforms)

# links

# we can also add links
print()
print("We can link ClassNodes manually:")
ssd.link(ClassNode("Person"), ClassNode("Address"), relationship="lives-at")
ssd.link(ClassNode("Address"), ClassNode("Person"), relationship="owned-by")
print(ssd)

print()
print("Though note that we can only choose links from the ontology:")
try:
    ssd.link(ClassNode("Person"), ClassNode("Address"), relationship="non-existing-link")
except Exception as err:
    print("Error: {0}".format(err))
print()

# samples...
print()
print("You can also sample columns")
print(ssd.sample(Column("address")))
print()
print("You can add transforms to samples as a preview")
print(ssd.sample(Column("address"), Transform(6)))

# we can also try to auto-fill the SSD
ssd.predict()
print()
print("Predicted model...")
print(ssd)

# we can just show the predicted models too
print()
print("Show just the predictions...")
for p in ssd.predictions:
    print(p)

# we can then correct it
print()
print("Now we can correct the model manually if necessary")
ssd.map(Column("LastName"), DataNode("Person", "last-name"))
print(ssd)

# we can also remove the elements
print()
print("We can remove links with lookups...")
ssd.remove(Link("owned-by"))
print(ssd)

# list all the data nodes
print()
print("Listing data nodes...")
for dn in ssd.data_nodes:
    print(dn)


# we can also remove the elements
# print()
# print("We can remove nodes with DataNode lookups...")
# ssd.remove(DataNode("Person", "last-name"))
# print(ssd)

# list all the data nodes
print()
print("Listing data nodes...")
for dn in ssd.data_nodes:
    print(dn)

# display
print()
print("We can also display the semantic model...")
ssd.show()

# save
print()
print("We can also save the semantic source description...")
ssd.save(file="out.ssd")

# we can access the model:
print()
print("The semantic model")
print(ssd.model)

print()
print("The semantic model links")
print(ssd.model.links)

print()
print("The semantic model nodes")
print(ssd.model.class_nodes)

# Or print out the json
print()
print("The source description JSON:")
print(ssd.json)

# Or print out the json
print()
print("The source description dictionary:")
print(ssd.ssd['semanticModel'])

print()
print("Columns")
print(ssd.columns)

print()
print("Mappings")
print(ssd.mappings)
