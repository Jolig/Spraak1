import numpy as np
import pandas as pd
import os
import pickle

import asr_evaluation as eval

mfcc_test_path = "./features/mfcc/test/timit.hdf"
mfcc_delta_test_path = "./features/mfcc_delta/test/timit.hdf"
mfcc_delta_delta_test_path = "./features/mfcc_delta_delta/test/timit.hdf"

dump_file_path = "./models"

delta_list = [0, 1, 2]
test_path_list = [mfcc_test_path, mfcc_delta_test_path, mfcc_delta_delta_test_path]



def main(delta, energy_coeff, models_path):

    test_features, test_labels = extracting_features_labels(delta, energy_coeff)
    models_dict = load_models(models_path)

    fla, per = testing(test_features, test_labels, models_dict)
    print(models_path, " ===== ", fla, "  ", per*100)



def extracting_features_labels(delta, energy_coeff):

    df = pd.read_hdf(test_path_list[int(delta)])

    features = np.array(df["features"].tolist())
    labels = np.array(df["labels"].tolist())


    if(energy_coeff is False):

        if (delta == '0'):
            features = np.delete(features, 0, axis=1)

        if (delta == '1'):
            features = np.delete(features, [0, 13], axis=1)

        if (delta == '2'):
            features = np.delete(features, [0, 13, 26], axis=1)

    return features, labels



def load_models(path):

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model



def testing(test_features, test_labels, models_dict):

    scores = []

    for label, model in models_dict.items():
        scores.append(model.score_samples(test_features))

    scores = np.array(scores)
    pred_indices = scores.argmax(axis = 0)

    pred_labels = []
    labels_list = list(models_dict.keys())

    for idx in pred_indices:
        pred_labels.append(labels_list[idx])

    return get_accuracy(test_labels, np.array(pred_labels)), eval.main(pred_labels, test_labels.tolist())



def get_accuracy(test_labels, pred_labels):

    common_list = [i for i, j in zip(test_labels, pred_labels) if i == j]
    cmp_perct = (len(common_list) / len(pred_labels)) * 100

    return cmp_perct



if __name__ == '__main__':

    print("              Model                          FLA                   PER")

    for dirName, subdirList, fileList in os.walk(dump_file_path):

        for mName in fileList:

            if(mName[0] != '.'):

                split = mName.split("_")

                delta = split[0]
                energy_coeff = True

                if (split[-1] == 'noEC.pkl'):
                    energy_coeff = False

                main(delta, energy_coeff, dirName+ "/"+mName)