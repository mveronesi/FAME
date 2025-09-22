# run experiment A.2 and B
from fame.experiments import exp_A_2_no_overwrite, exp_B_no_overwrite, exp_C_no_overwrite
from keras.models import load_model
from keras.datasets import mnist
import pandas as pd
import os
import numpy as np
import pickle
from fame.abstract_domain.utils import check_is_robust 


# run in shell:
# for i in {1..62}; do echo "Run $i"; python script_gtsrb_10x2.py; done


def get_model(MODEL):
    return load_model('./models/xairobas_gtsrb-{}.keras'.format(MODEL))

def get_data():
    """
    download and process GTSRB data.
    """
    filename = "gtsrb.pickle"
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    x_test, y_test = data['x_test'], data['y_test']
    x_test = x_test.astype('float32') / 255
    x_test = np.reshape(x_test, (-1, 3072))

    return x_test, y_test

def get_indices():

    return [i for i in range(100) if not i in [
    1, 2, 3, 4, 8, 11, 16, 20, 22, 24, 26, 31, 42, 43, 44, 46, 49, 50, 53, 55,
    56, 57, 58, 64, 67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 87, 92, 93, 94,
    1, 2, 3, 4, 8, 11, 16, 20, 22, 24, 26, 31, 42, 43, 44, 46, 49, 50, 53, 55,
    56, 57, 58, 64, 67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 87, 92, 93, 94,
    ]]


def func_gtsrb_10x2_expA2():
    DATASET='GTSRB'
    MODEL='10x2'
    eps=0.01

    channel=3
    data_format="channels_last"
    n_class=10
    ## load the model and data

    indices = get_indices()
    k_model = get_model(MODEL)
    x_test, y_test = get_data()

    def is_robust(j):
        return check_is_robust(model=k_model, 
                        input_sample=x_test[j], 
                        eps=eps, 
                        channel=channel, 
                        data_format=data_format, 
                        n_class=n_class)
    indices = [i for i in indices if not is_robust(i)]

    
    dataframe_repository='./results'
    EXP="A_2"
    filename = "{}_{}_{}".format(DATASET, MODEL, EXP)
    dataframe_filename = filename

    if os.path.isfile("{}/{}.csv".format(dataframe_repository, dataframe_filename)):
        df_before = pd.read_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename))
        i = len(df_before.index)
    else:
        i=0

    exp_A_2_no_overwrite(
        model=k_model,
        x_test=x_test,
        y_test=y_test,
        indices=[indices[i]],
        eps=eps,
        dataframe_repository=dataframe_repository,
        dataframe_filename=dataframe_filename,
        channel=channel,
        data_format=data_format,
        n_class = n_class,
    )

def func_gtsrb_10x2_expB():

    DATASET='GTSRB'
    MODEL='10x2'
    eps=0.01

    channel=3
    data_format="channels_last"
    n_class=10
    ## load the model and data

    indices = get_indices()
    k_model = get_model(MODEL)
    x_test, y_test = get_data()

    def is_robust(j):
        return check_is_robust(model=k_model, 
                        input_sample=x_test[j], 
                        eps=eps, 
                        channel=channel, 
                        data_format=data_format, 
                        n_class=n_class)
    indices = [i for i in indices if not is_robust(i)]

    dataframe_repository='./results'
    EXP="B"
    filename = "{}_{}_{}".format(DATASET, MODEL, EXP)
    dataframe_filename = filename

    if os.path.isfile("{}/{}.csv".format(dataframe_repository, dataframe_filename)):
        df_before = pd.read_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename))
        i = len(df_before.index)
    else:
        i=0

    exp_B_no_overwrite(
        model=k_model,
        x_test=x_test,
        y_test=y_test,
        indices=[indices[i]],
        eps=eps,
        dataframe_repository=dataframe_repository,
        dataframe_filename=dataframe_filename,
        channel=channel,
        data_format=data_format,
        method="greedy",
        attack="fgsm",
        traversal_order="greedy",
        device="mps",
        n_class = n_class,
    )

def func_gtsrb_10x2_expC():

    DATASET='GTSRB'
    MODEL='10x2'
    eps=0.01

    channel=3
    data_format="channels_last"
    n_class=10
    ## load the model and data

    indices = get_indices()
    k_model = get_model(MODEL)
    x_test, y_test = get_data()

    def is_robust(j):
        return check_is_robust(model=k_model, 
                        input_sample=x_test[j], 
                        eps=eps, 
                        channel=channel, 
                        data_format=data_format, 
                        n_class=n_class)
    indices = [i for i in indices if not is_robust(i)]

    dataframe_repository='./results'
    EXP="C"
    filename = "{}_{}_{}".format(DATASET, MODEL, EXP)
    dataframe_filename = filename

    if os.path.isfile("{}/{}.csv".format(dataframe_repository, dataframe_filename)):
        df_before = pd.read_csv("{}/{}.csv".format(dataframe_repository, dataframe_filename))
        i = len(df_before.index)
    else:
        i=0

    exp_C_no_overwrite(
        model=k_model,
        x_test=x_test,
        y_test=y_test,
        indices=[indices[i]],
        eps=eps,
        dataframe_repository=dataframe_repository,
        dataframe_filename=dataframe_filename,
        channel=channel,
        data_format=data_format,
        method="greedy",
        attack="fgsm",
        traversal_order="greedy",
        device="mps",
        n_class = n_class,
    )

if __name__ == "__main__":
    func_gtsrb_10x2_expC()
