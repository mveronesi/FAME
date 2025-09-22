# run experiment A.2 and B
from fame.experiments import exp_A_1, exp_A_2, exp_A_2_no_overwrite, exp_B_no_overwrite, exp_C_no_overwrite
from keras.models import load_model
from keras.datasets import mnist
import pandas as pd
import os
import numpy as np
from fame.abstract_domain.utils import check_is_robust 


# run in shell:
# for i in {1..67}; do echo "Run $i"; python script_mnist_cnn.py; done

def get_model(MODEL):
    return load_model('./models/xairobas_mnist-{}.keras'.format(MODEL))

def get_data():
    """
    download and process MNIST data.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    X = np.concatenate([x_train, x_test])
    Y = np.concatenate([y_train, y_test])

    return X, Y

def get_indices():
    return [0, 2, 3, 4, 5, 6, 8, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 89, 92, 93, 94, 96, 97, 98, 99]

def func_mnist_cnn_expA2():
    DATASET='MNIST'
    MODEL='cnn'
    eps=0.05

    channel=1
    data_format="channels_first"
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

def func_mnist_cnn_expB(attack="fgsm"):
    DATASET='MNIST'
    MODEL='cnn'
    eps=0.05

    channel=1
    data_format="channels_first"
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
    filename = "{}_{}_{}_{}".format(DATASET, MODEL, EXP, attack)
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
        attack=attack,
        traversal_order="greedy",
        device="mps",
        n_class = n_class,
    )

def func_mnist_cnn_expC(attack="fgsm"):
    DATASET='MNIST'
    MODEL='cnn'
    eps=0.05

    channel=1
    data_format="channels_first"
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
    filename = "{}_{}_{}_{}".format(DATASET, MODEL, EXP, attack)
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
        attack=attack,
        traversal_order="greedy",
        device="mps",
        n_class = n_class,
    )

if __name__ == "__main__":
    func_mnist_cnn_expC()
