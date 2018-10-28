import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture



mfcc_train_path = "./features/mfcc/train/timit.hdf"
mfcc_delta_train_path = "./features/mfcc_delta/train/timit.hdf"
mfcc_delta_delta_train_path = "./features/mfcc_delta_delta/train/timit.hdf"

delta_list = [0, 1, 2]
train_path_list = [mfcc_train_path, mfcc_delta_train_path, mfcc_delta_delta_train_path]

dump_file_path = "./models"



def main(caseID, delta, energy_coeff, num_mixtures):

    training_dict = constructing_training_dict(delta, energy_coeff)

    models_dict = training(training_dict, num_mixtures)

    dump_models(caseID, models_dict, delta, energy_coeff, num_mixtures)



def constructing_training_dict(delta, energy_coeff):

    df = pd.read_hdf(train_path_list[delta])

    features = np.array(df["features"].tolist())
    labels = np.array(df["labels"].tolist())


    if(energy_coeff is False):

        if(delta == 0):
            features = np.delete(features, 0, axis = 1)

        if(delta == 1):
            features = np.delete(features, [0, 13], axis = 1)

        if(delta == 2):
            features = np.delete(features, [0, 13 , 26], axis = 1)


    unique_labels = np.unique(labels)
    training_dict = {key: [] for key in unique_labels}

    for label, feature in zip(labels, features):
        training_dict[label].append(feature)

    for label, features in training_dict.items():
        training_dict[label] = np.array(training_dict[label])

    return training_dict



def training(training_dict, num_mixtures):

    models_dict = {}

    for label, features in training_dict.items():
        model = GaussianMixture(n_components = int(num_mixtures),
                                covariance_type = "diag")
        model.fit(features)
        models_dict[label] = model

    return models_dict



def dump_models(caseID, models_dict, delta, energy_coeff, num_mixtures):

    path =  "/Case " + caseID + "/" + str(delta) + "_" + num_mixtures+ "_"

    if(energy_coeff is True):
        path = path + "EC"
    else:
        path = path + "noEC"

    path = dump_file_path + path + ".pkl"

    with open(path, 'wb') as f:
        pickle.dump(models_dict, f)



if __name__ == '__main__':

    for delta in delta_list:
        main('1', delta, True, '064')

    for delta in delta_list:
        main('2', delta, False, '064')


    num_mixtures_list = ['002', '004', '008', '016', '032', '064', '128', '256']

    for num_mixtures in num_mixtures_list:
        main('3', 0, False, num_mixtures)





