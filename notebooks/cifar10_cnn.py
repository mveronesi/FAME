import sys
sys.path.append("../")
import os
os.environ["KERAS_BACKEND"] = "torch"

from fame.experiments import exp_A_1, exp_A_2, exp_A_2_no_overwrite, exp_B_no_overwrite
from keras.models import load_model
import numpy as np
from argparse import ArgumentParser, Namespace

# check robustness because of numerical approximation error between solvers
from fame.abstract_domain.utils import check_is_robust 

from configs.cifar10_configs import get_dataset, dataset_to_numpy, means_np, stddevs_np
import random

random.seed(42)


def main(args: Namespace):
    DATASET = "CIFAR10"
    MODEL = "cnn"
    method = args.method.lower()
    if method not in {"milp", "greedy"}:
        raise ValueError("--method must be either 'milp' or 'greedy'")

    means_avg = np.mean(means_np)
    std_avg = np.mean(stddevs_np)
    eps = args.eps
    norm = args.norm

    print("eps:", eps)
    channel = 3
    data_format = "channels_last"
    n_class = 10
    
    print("norm:", norm)
    print("method:", method)

    """
    download and process CIFAR10 data.
    """
    test_dataset = get_dataset(augment=False, get_train=False, get_val=False)
    x_test, y_test = dataset_to_numpy(test_dataset, means_np, stddevs_np)
    x_test_flattened = np.reshape(x_test, (-1, 3072))
    print(f"x_test shape (Normalisé, NHWC): {x_test.shape}")
    print(f"x_test dtype: {x_test.dtype}")

    k_model = load_model("./models/resnet_2b_ported.keras")
    k_model.eval()

    robust_eps003 = [6,  13,  15,  16,  19,  21,  23,  29,  34,  41,  44,  45,  50,  54,
            60,  73,  75,  79,  82,  84,  90,  92,  98,  99, 102, 103, 104, 105,
            116, 122, 123, 131, 133, 141, 142, 151, 153, 154, 157, 166, 175, 196,
            202, 204, 208, 209, 215, 216, 220, 222, 225, 231, 232, 235, 240, 243,
            244, 252, 257, 265, 272, 276, 280, 283, 285, 286, 288, 289, 290, 296,
            297, 298, 311, 315, 321, 330, 331, 333, 334, 338, 341, 344, 345, 348,
            353, 361, 362, 369, 371, 372, 374, 379, 381, 382, 386, 389, 390, 392,
            400, 406, 414, 415, 417, 425, 429, 431, 440, 442, 443, 447, 452, 460,
            462, 469, 471, 472, 475, 484, 486, 487, 489, 490, 493, 495, 497, 499,
            507, 510, 511, 512, 514, 516, 517, 519, 521, 523, 524, 527, 529, 533,
            540, 541, 542, 544, 546, 547, 560, 571, 572, 576, 581, 588, 590, 591,
            592, 600, 601, 604, 605, 608, 609, 610, 612, 613, 619, 622, 626, 643,
            654, 656, 662, 664, 666, 681, 691, 693, 696, 700, 709, 721, 723, 724,
            726, 732, 736, 738, 741, 743, 745, 747, 750, 753, 759, 763, 764, 765,
            772, 774, 782, 786, 789, 791, 800, 801, 803, 812, 813, 815, 823, 824,
            827, 828, 830, 832, 838, 839, 840, 842, 844, 847, 854, 856, 857, 858,
            859, 864, 868, 871, 872, 874, 879, 883, 885, 891, 894, 899, 902, 903,
            911, 914, 915, 917, 921, 925, 931, 934, 935, 939, 940, 946, 951, 955,
            958, 959, 960, 968, 973, 975, 984, 985, 989, 990, 994, 997, 999]

    #robust_eps003= [  6,  13,  15,  16,  19,  21,  23,  29,  34,  41,  44,  45,  50,  54, 60,  73,  75,  79,  82,  84,  90,  92,  98,  99, ]
    indices = [i for i in range(0,1000) if i not in robust_eps003]
    indices = indices[:100]
    random.shuffle(indices)
    indices = [24]
    print("Indices:", indices)

    dataframe_repository = "./results/CIFAR10/eps-{}".format(f'{eps}'.replace('0.', ''))
    os.makedirs(dataframe_repository, exist_ok =True)

    EXP = "A_2"
    filename = "{}_{}_{}_{}".format(DATASET, MODEL, EXP, method)
    dataframe_filename_A2 = filename

    failing_indexes_a2 = []
    for i, j in enumerate(indices):
        print(f"Processing index {j} ({i+1}/{len(indices)})...")
        try:
            dico_a_2 = exp_A_2_no_overwrite(
                model=k_model,
                x_test=x_test_flattened,
                y_test=y_test,
                indices=[j],#indices,
                eps=eps,
                dataframe_repository=dataframe_repository,
                dataframe_filename=dataframe_filename_A2,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                verbose=2,
                sleep_time=0,
                means=means_avg,
                stddev=std_avg,
                norm=norm,
                method=method,
            )
        except Exception as ex:
            print("exception: ", ex)
            failing_indexes_a2.append(j)
            print("fail: ", failing_indexes_a2)
            continue

    # EXP = "A_1"
    # filename = "{}_{}_{}".format(DATASET, MODEL, EXP)
    # dataframe_filename_A1 = filename

    # dico_a_1 = exp_A_1(
    #         model=k_model,
    #         x_test=x_test_flattened,
    #         y_test=y_test,
    #         indices=indices, #new_indices
    #         eps=eps,
    #         dataframe_repository=dataframe_repository,
    #         dataframe_filename=dataframe_filename_A1,
    #         channel=channel,
    #         data_format=data_format,
    #         n_class=n_class,
    #         verbose=1,
    #         means=means_avg,
    #         stddev=std_avg,
    #         norm=norm,
    #         #decomon_method=decomon_method
    #     )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eps", required=True, type=float, help="Perturbation size for adversarial attacks.")
    parser.add_argument(
        "--norm",
        required=False,
        type=float,
        default=np.inf,
        help="Norm type for adversarial attacks (default: 2).",
    )
    parser.add_argument("--method", required=False, type=str, default="greedy", help="Search method for experiment A (milp or greedy).")
    args = parser.parse_args()
    main(args)
