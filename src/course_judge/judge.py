import numpy as np

from src.enums import Fields
from src.ml.MLMetrics import MLMetrics
from src.ml.NaiveBayes import NaiveBayes
from src.models.Manager import Manager
from src.models.TextPreprocessor import EnglishTextPreprocessor
from src.utils.read_document import read_documents_json

training_data_path = "../../MIR_Phase2/data/train.json"
test_data_path = "./test_1000.json"

smoothing = 1
num_classes = 4
text_preprocessor = EnglishTextPreprocessor(stem=False, lemmatize=True)
train_documents = read_documents_json(training_data_path, text_preprocessor)
val_documents = read_documents_json(test_data_path, text_preprocessor)
manager = Manager(train_documents, [Fields.BODY, Fields.TITLE], text_preprocessor)
train_labels = np.array([doc.category - 1 for doc in train_documents])
val_labels = np.array([doc.category - 1 for doc in val_documents])

clf = NaiveBayes(manager, smoothing, num_classes)
clf.train()
results = np.array(clf.test(val_documents))

cm = MLMetrics.compute_confusion_matrix(val_labels, results)
accuracy = MLMetrics.accuracy(cm)

print("Accuracy: {:.3f}".format(accuracy))
