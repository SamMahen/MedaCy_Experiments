from medacy.data import Dataset
from medacy.pipelines import SystematicReviewPipeline
from medacy.model import Model
from medacy.pipeline_components import MetaMap
import logging,sys



# print logs
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

#entity types
entities = ['ADR', 'Indication', 'Drug']

# training_dataset, evaluation_dataset, meta_data = Dataset.load_external('medacy_dataset_smm4h_2019')
training_dataset = Dataset('/home/mahendrand/VE/SMM4H/data_smmh4h/task2/training/dataset')
#path = '../data_smmh4h/task2/training/dataset_1'
#set metamap path
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap", convert_ascii=True)
training_dataset.metamap(metamap)

# pipeline = SystematicReviewPipeline(metamap=None, entities=meta_data['entities'])
pipeline = SystematicReviewPipeline(metamap=metamap, entities=entities)
model = Model(pipeline, n_jobs=1) #distribute documents between 30 processes during training and prediction

model.fit(training_dataset)
model.cross_validate(num_folds = 5, dataset = training_dataset, write_predictions=True)

#location to store the clinical model
model.dump('/home/mahendrand/VE/SMM4H/medaCy/medacy/clinical_model.pickle')

#location to store the predictions
#model.predict(training_dataset, prediction_directory='/home/mahendrand/VE/SMM4H/data_smmh4h/task2/training/dataset/metamap_predictions')