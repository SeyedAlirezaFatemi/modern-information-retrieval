import json

import model

training_data_path = "../../MIR_Phase2/data/train.json"
test_data_path = "test.json"

with open(training_data_path) as training_data_file:
    training_data = json.loads(training_data_file.read())

with open(test_data_path) as test_data_file:
    test_data = json.loads(test_data_file.read())

model.train(training_data)

corrects = 0
total = len(test_data)

for doc in test_data:
    category = doc.pop("category")
    predicted_category = model.classify(doc)
    if category == predicted_category:
        corrects += 1

accuracy = corrects / total

print("Accuracy: {:.3f}".format(accuracy))
