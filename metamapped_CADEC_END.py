from medacy.data import Dataset
from medacy.ner.pipelines import ClinicalPipeline
from medacy.ner.model import Model
from medacy.pipeline_components import MetaMap

import logging,sys

# print logs
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

#set metamap path
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap", convert_ascii=True)

#entity types
entities = ['Reason', 'ADE', 'Drug']

pipeline = ClinicalPipeline(metamap=None, entities=entities)


# fold1
# training_dataset_1 = Dataset('/home/mahendrand/VE/Data/CADEC_END/1/train')
# # training_dataset_1.metamap(metamap)
#
# model_1 = Model(pipeline, n_jobs=1)
# model_1.fit(training_dataset_1)

# #run on a separate testing dataset
# testing_dataset_1= Dataset('/home/mahendrand/VE/Data/CADEC_END/1/test')
# # location to store the predictions
# model.predict(testing_dataset_1, prediction_directory='/home/mahendrand/VE/Data/preds/5 fold/CADEC_END')
#
#
# #fold 2
training_dataset_2 = Dataset('/home/mahendrand/VE/Data/CADEC_END/2/train')
# training_dataset_2.metamap(metamap)
#
model_2 = Model(pipeline, n_jobs=1)
model_2.fit(training_dataset_2)

#run on a separate testing dataset
testing_dataset_2= Dataset('/home/mahendrand/VE/Data/CADEC_END/2/test')
# location to store the predictions
model.predict(testing_dataset_2, prediction_directory='/home/mahendrand/VE/Data/preds/5 fold/CADEC_END')
#
#
# #fold 3
# training_dataset_3 = Dataset('/home/mahendrand/VE/Data/CADEC_END/3/train')
# training_dataset_3.metamap(metamap)
#
# model_3 = Model(pipeline, n_jobs=1)
# model_3.fit(training_dataset_3)
#
# #run on a separate testing dataset
# testing_dataset_3= Dataset('/home/mahendrand/VE/Data/CADEC_END/3/test')
# # location to store the predictions
# model.predict(testing_dataset_3, prediction_directory='/home/mahendrand/VE/Data/preds/5 fold/CADEC_END')
#
#
# #fold 4
# training_dataset_4 = Dataset('/home/mahendrand/VE/Data/CADEC_END/4/train')
# training_dataset_4.metamap(metamap)
#
# model_4 = Model(pipeline, n_jobs=1)
# model_4.fit(training_dataset_4)
#
# #run on a separate testing dataset
# testing_dataset_4= Dataset('/home/mahendrand/VE/Data/CADEC_END/4/test')
# # location to store the predictions
# model.predict(testing_dataset_4, prediction_directory='/home/mahendrand/VE/Data/preds/5 fold/CADEC_END')
#
#
# #fold 5
# training_dataset_5 = Dataset('/home/mahendrand/VE/Data/CADEC_END/5/train')
# training_dataset_5.metamap(metamap)
#
# model_5 = Model(pipeline, n_jobs=1)
# model_5.fit(training_dataset_5)
#
# #run on a separate testing dataset
# testing_dataset_5= Dataset('/home/mahendrand/VE/Data/CADEC_END/5/test')
# # location to store the predictions
# model.predict(testing_dataset_5, prediction_directory='/home/mahendrand/VE/Data/preds/5 fold/CADEC_END')
