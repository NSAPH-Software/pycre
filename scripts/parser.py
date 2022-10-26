import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument("--seed", default=1, type=int, help="seed")

    # dataset
    parser.add_argument("--dataset_name", default="syntethic", type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="", type=str, help="dataset directory")
    parser.add_argument("--standardize", default=False, type=bool, help="whether standardize or not the covariates")
    parser.add_argument("--subsample", default=1, type=float, help="ratio of the dataset to subsample")
    # synthtich dataset
    parser.add_argument("--n", default=100, type=int, help="number of observations for the Syntetic Dataset")
    parser.add_argument("--k", default=10, type=int, help="number of covariates for the Syntetic Dataset")
    parser.add_argument("--binary", default=True, type=bool, help="whether the outcome is binary or continuos for the Syntetic Dataset")
    parser.add_argument("--effect_size", default=2, type=float, help="effect size maginitude for the (continuos outcome) Syntetic Dataset")
    #splitting
    parser.add_argument("--ratio_dis", default=0.5, type=float, help="ratio of the observations used for discovery")

    # ITE estimation

    # CATE estimation
    
    return parser