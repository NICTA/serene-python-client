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
from serene import DataNode, ClassNode

STOPFORINPUT = True

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

data_path = os.path.join(os.path.dirname(__file__), "resources", "data")
owl_path = os.path.join(os.path.dirname(__file__), "resources", "owl")

# get frist model
octo = sn.octopii.items[0]

# =======================
#
#  Run predictions
#
# =======================

test_predict = sn.datasets.upload(os.path.join(data_path, 'predict1.csv'))
#test_predict = sn.datasets.upload(os.path.join(data_path, 'predict2.csv'))
#test_predict = sn.datasets.upload(os.path.join(data_path, 'train1.csv'))

predicted = octo.predict(test_predict)

print()
print('Predicted schema mathcer results::')
print()
schema_results = octo.matcher_predict(test_predict)
print(schema_results)

print()
print('Predicted results:: {} results'.format(len(predicted)))
print()
for prediction in predicted[:3]:
    print(prediction.score)
    prediction.ssd.show()
    if STOPFORINPUT:
        input('Press enter to continue...')

# =======================
#
#  Confirm the result and add to the training data on the server
#
# =======================

predicted_ssd = predicted[0].ssd
predicted_ssd.show()
new_ssd = sn.ssds.upload(predicted_ssd)
octo = sn.octopii.update(octo.add(new_ssd))

# =======================
#
#  Extrac the mappings
#
# =======================

person_attributes = ['birthDate', 'gender', 'identifier']
for attr in person_attributes:
    maps = octo.mappings(DataNode(ClassNode('Person'), attr))
    print()
    print('These are the columns mapped to Person.{}'.format(attr))
    for dn in maps:
        print(dn, 'which is in', dn.dataset)
    print()

maps = octo.mappings(DataNode(ClassNode('Address'), 'identifier'))
print()
print('These are the columns mapped to Address.identifier')
for dn in maps:
    print(dn, 'which is in', dn.dataset)
print()

maps = octo.mappings(DataNode(ClassNode('Phone'), 'identifier'))
print()
print('These are the columns mapped to Phone.identifier')
for dn in maps:
    print(dn, 'which is in', dn.dataset)
print()

maps = octo.mappings(DataNode(ClassNode('Email'), 'identifier'))
print()
print('These are the columns mapped to Email.identifier')
for dn in maps:
    print(dn, 'which is in', dn.dataset)
print()
