from medacy.data import Dataset
import logging,sys

from pprint import pprint

# print logs
# logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

#entity types
# entities = ['CellLine','Dose','DoseDuration', 'DoseDurationUnits', 'DoseFrequency', 'DoseRoute', 'DoseUnits', 'Endpoint', 'EndpointUnitOfMeasure', 'GroupName', 'GroupSize', 'SampleSize', 'Sex', 'Species', 'Strain', 'TestArticle', 'TestArticlePurity', 'TestArticleVerification', 'TimeAtDose', 'TimeAtFirstDose', 'TimeAtLastDose', 'TimeEndpointAssessed', 'TimeUnits', 'Vehicle' ]

# training_dataset, evaluation_dataset, meta_data = Dataset.load_external('medacy_dataset_smm4h_2019')
training_dataset = Dataset('/home/mahendrand/VE/TAC/data_TAC')
prediction_dataset = Dataset('/home/mahendrand/VE/TAC/data_TAC/predictions')



ambiguity_dict = training_dataset.compute_ambiguity(prediction_dataset)
#pprint(ambiguity_dict)

entities, confusion_matrix = training_dataset.compute_confusion_matrix(prediction_dataset, leniency=1)

pprint(training_dataset.compute_counts())

print(entities)
pprint(confusion_matrix)