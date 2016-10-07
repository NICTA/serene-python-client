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

import dataint
import pandas as pd

from dataint import SemanticModeller, Ontology, DataNode, Link, Column, ClassNode

# start with a SemanticModeller object...
sm = dataint.SemanticModeller(
    config={
        "kmeans": 5,
        "threshold": 0.7,
        "search-depth": 100
    },
    ontologies=["/some/file/something.owl", "/some/other/file/something.owl"]
)
print(sm)


# build up a simple ontology
on = (dataint.Ontology()
      .prefix("asd", "asd:/9e8rgyusdgfiuhsdf.dfguhudfh")
      .uri("http://www.semanticweb.org/data_integration_project/report_example_ontology")
      .class_node("Person", ["name", "phone"])
      .class_node("Location", prefix="xsd")
      .class_node("City", is_a="Location")
      .class_node("State", is_a="Location")
      .class_node("Address", {"name": str, "postcode": int}, is_a="Location")
      .relationship("Person", "lives-at", "Address")
      .relationship("Address", "is_in", "City")
      .relationship("Address", "is_in", "State"))

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
ssd = sm.to_ssd("example/resources/data/EmployeeAddresses.csv")

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
ssd.map(Column("postcode"), DataNode("postcode"))

print()
print("Adding one map...")
print(ssd)

# you can assign the DataNodes with a transform...
ssd.map(Column("FirstName"), DataNode("name"), transform=pd.to_datetime)
print()
print("Adding a map with a transform")
print(ssd)
