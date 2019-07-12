from medacy.data import Dataset
from medacy.ner.pipelines import ClinicalPipeline
from medacy.ner.model import Model
from medacy.pipeline_components import MetaMap

import logging,sys
import os
import shutil

# print logs
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG) #set level=logging.DEBUG for more information

# information from the user
folds = 5
dataset1 = '/home/mahendrand/VE/Data/N2C2/data'
dataset2 = '/home/mahendrand/VE/Data/END/drug'
dirTrain = '/home/mahendrand/VE/Data/MultiModel/END_N2C2/train'
dirTest = '/home/mahendrand/VE/Data/MultiModel/END_N2C2/test'
dirPrediction = '/home/mahendrand/VE/Predictions/multi_fold/END_N2C2'


#entity types
entities = ['Reason', 'ADE', 'Drug']

#set metamap path
metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap", convert_ascii=True)

# build the pipeline
pipeline = ClinicalPipeline(metamap=None, entities=entities)

def get_files(path):
    files = os.listdir(path)
    files.sort()
    ann_1 = []
    txt_1 = []
    for f in files:
        if f.rsplit('.', 1)[-1]== 'ann':
            ann_1.append(f)
        if f.rsplit('.', 1)[-1]== 'txt':
            txt_1.append(f)
    return ann_1, txt_1

def create_directory(dirName):

    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        # Delete the existing and create target Directory if it exists
        shutil.rmtree(dirName)
        os.mkdir(dirName)
        print("Directory " , dirName ,  "Created again")

ann_files_1, txt_files_1 = get_files(dataset1)
ann_files_2, txt_files_2 = get_files(dataset2)

num_files = int(len(ann_files_1) / folds)

for i in range(folds):
    create_directory(dirTest)
    create_directory(dirTrain)
    print("Fold : ",i)

    for item in ann_files_1:
        shutil.copy(dataset1 + '/' + item, dirTrain)
    for item in ann_files_2:
        shutil.copy(dataset2 + '/' + item, dirTrain)
    for item in txt_files_1:
        shutil.copy(dataset1 + '/' + item, dirTrain)
    for item in txt_files_2:
        shutil.copy(dataset2 + '/' + item, dirTrain)

    for item in ann_files_1[i * num_files:(i + 1) * num_files]:
        shutil.copy(dataset1 + '/' + item, dirTest)
        os.remove(dirTrain + '/' + item)
    for item in txt_files_1[i * num_files:(i + 1) * num_files]:
        shutil.copy(dataset1 + '/' + item, dirTest)
        os.remove(dirTrain + '/' + item)

    training_dataset = Dataset(dirTrain)
    training_dataset.metamap(metamap)

    model = Model(pipeline, n_jobs=1)
    model.fit(training_dataset)

    # run on a separate testing dataset
    testing_dataset = Dataset(dirTest)

    # location to store the predictions
    model.predict(testing_dataset, prediction_directory = dirPrediction)