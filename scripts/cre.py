import logging

from parser import get_parser
from dataset import get_dataset
from ite import estimate_ite_ipw
from utils import standardize
from decision_rules import generate_rules, get_rules_matrix, rules_regularization
from cate import estimate_cate

import numpy as np
import pandas as pd

def CRE(dataset, args):

    # 1. Discovery
    print(f"- Discovery Step:")
    
    # Split Dataset
    args.n = dataset.shape[0]
    n_dis = int(args.n*args.ratio_dis)

    dataset_dis = dataset.iloc[:,:n_dis]
    y_dis = dataset_dis["y"]
    t_dis = dataset_dis["t"]
    X_dis = dataset_dis.drop(['y', 't'], axis=1)

    dataset_inf = dataset.iloc[n_dis:,:]
    y_inf = dataset_inf["y"]
    t_inf = dataset_inf["t"]
    X_inf = dataset_inf.drop(['y', 't'], axis=1)

    # Esitimate ITE
    print(f"    ITE Estimation")
    ite_dis = estimate_ite_ipw(X = X_dis, 
                               y = y_dis, 
                               t = t_dis)
    ite_dis_std = standardize(ite_dis)

    # Rules Generation
    print(f"    (Causal) Rules Generation")
    rules = generate_rules(X = X_dis, 
                           ite = ite_dis_std)
    R_dis = get_rules_matrix(rules, X_dis)

    # Rules Regularization
    print(f"    (Causal) Rules Regularization")
    rules = rules_regularization(R_dis)

    # 2. Inference
    print(f"- Inference Step:")
    # Esitimate ITE
    print(f"    ITE Estimation")
    ite_inf = estimate_ite_ipw(X = X_inf, 
                               y = y_inf, 
                               t = t_inf)
    print(f"    CATE estimatation")
    R_inf = get_rules_matrix(rules, X_inf)
    #R_inf.to_csv("results/R_inf.csv")
    CATE = estimate_cate(R_inf, ite_inf)
    print(CATE.summary())
    return

def main(args):
    # reproducibility
    np.random.seed(args.seed)

    print(f"Load {args.dataset_name} dataset")
    dataset = get_dataset(args)
    
    print(f"Run CRE algorithm")
    result = CRE(dataset, args)

    return result

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
