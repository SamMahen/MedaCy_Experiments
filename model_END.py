from medacy.data import Dataset
# from medacy.ner.pipelines import SystematicReviewPipeline
from medacy.ner.pipelines import ClinicalPipeline
from medacy.ner.model import Model
import logging,sys

#logging.basicConfig(filename=model_directory+'/build_%cd .log' % current_time,level=logging.DEBUG) #set level=logging.DEBUG for more information
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

# entities = ['Form','Route','Frequency', 'Reason', 'Duration', 'Dosage', 'ADE', 'Strength', 'Drug' ]
entities = ['Symptom', 'Drug' ]

# training_dataset, evaluation_dataset, meta_data = Dataset.load_external('medacy_dataset_smm4h_2019')
training_dataset = Dataset('/home/mahendrand/VE/Data/END/symptom')


#training_dataset.set_data_limit(10)
# pipeline = SystematicReviewPipeline(metamap=None, entities=meta_data['entities'])
pipeline = ClinicalPipeline(metamap=None, entities=entities)
model = Model(pipeline, n_jobs=1) #distribute documents between 30 processes during training and prediction
#
model.fit(training_dataset)

model.cross_validate(num_folds = 5, training_dataset = training_dataset, prediction_directory=True, groundtruth_directory=True)


# model.dump('/home/mahendrand/VE/SMM4H/medaCy/medacy/clinical_model.pickle')
# model.predict(training_dataset, prediction_directory='/home/mahendrand/VE/data_smmh4h/task2/training/metamap_predictions')

# model.predict(training_dataset)


# train_dataset, evaluation_dataset, meta_data = Dataset.load_external('medacy_dataset_smm4h_2019')
#
# print(train_dataset)
