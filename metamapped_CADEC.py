from medacy.data import Dataset
from medacy.ner.pipelines import SystematicReviewPipeline
from medacy.ner.model import Model
from medacy.pipeline_components import MetaMap

import logging,sys

# print logs
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

#entity types
entities = ['Reason', 'ADE', 'Drug']

training_dataset = Dataset('/home/mahendrand/VE/Data/CADEC/converted')
# dirPred = '/home/mahendrand/VE/Data/preds/CV/CADEC'
#set metamap path
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap", convert_ascii=True)
training_dataset.metamap(metamap)


pipeline = SystematicReviewPipeline(metamap=metamap, entities=entities)
model = Model(pipeline, n_jobs=1) #distribute documents between 30 processes during training and prediction

model.fit(training_dataset)

#cross validation
# model.cross_validate(num_folds = 5, training_dataset = training_dataset, prediction_directory=dirPred, groundtruth_directory=True)

#location to store the clinical model
# model.dump('/home/mahendrand/VE/SMM4H/medaCy/medacy/clinical_model.pickle')

# run on a separate testing dataset
testing_dataset_END = Dataset('/home/mahendrand/VE/Data/END/drug')

# location to store the predictions
model.predict(testing_dataset_END, prediction_directory='/home/mahendrand/VE/Predictions/cross_domain/CADEC_END')

#run on a separate testing dataset
testing_dataset_CADEC = Dataset('/home/mahendrand/VE/Data/N2C2/data')

# location to store the predictions
model.predict(testing_dataset_CADEC, prediction_directory='/home/mahendrand/VE/Predictions/cross_domain/CADEC_N2C2')