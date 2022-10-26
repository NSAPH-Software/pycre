import pandas as pd
import numpy as np

def get_dataset(args):
    """
    Get Dataset

    Input
        args: Experiment arguments
    Output
        dataset: pd.DataFrame with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
    """
    if args.dataset_name == "syntethic":
        dataset = generate_syn_dataset(n = args.n, 
                                       k = args.k, 
                                       binary = args.binary,
                                       effect_size = args.effect_size)
    else: 
        raise ValueError("TO DO: check if path exists and read data")
    
    if args.standardize:
        raise ValueError("TO DO: standardize covariates")

    if args.subsample!=1:
        dataset = dataset.sample(frac=args.subsample)

    return dataset

def generate_syn_dataset(n = 100, k = 10, binary = True, 
                         effect_size = 2):
    """
    Generate a Syntethic Dataset

    Input
        n: number of observations
        k: number of covariates 
        binary: whether the outcome is binary or continuos
        effect_size: effect size magnitude

    Output
        dataset: pd.DataFrame with Covariates ('x1', ..., 'xk'), 
                 Treatments ('t') and Outcome ('y')
    """

    # Covariates
    X = np.random.uniform(low = 0, 
                          high = 1, 
                          size = (n, k))
    X_names = ["x"+str(i) for i in range(1,k+1)]
    dataset = pd.DataFrame(data = X, 
                           columns = X_names)

    # Treatment
    logit = -1 + 2*(dataset["x1"]-dataset["x2"]+dataset["x3"])
    prob = np.exp(logit) / (1 + np.exp(logit))
    t = np.random.binomial(n = 1,
                           p = prob, 
                           size = n)
    dataset['t'] = t
    
    # Outcome
    rule_a = dataset['x1']>0.7
    rule_b = dataset["x2"]>0.6
    rule_c = dataset["x2"]>0.3
    rule_d = dataset["x3"]<0.2

    if binary: 
        y0 = np.zeros(n)
        y0[rule_a|rule_b] = 1

        y1 = np.zeros(n)
        y1[rule_c|rule_d] = 1
        tau = y1 - y0
    else: 
        tau = np.zeros(n)
        tau[rule_a|rule_b] = effect_size
        tau[rule_c|rule_d] = - effect_size

        mu = -1 + 2*(dataset["x1"]-dataset["x2"]+dataset["x3"])
        y0 = np.random.normal(loc = mu,
                              scale = 1,
                              size = n)
        y1 = y0 + tau

    y = y0 * (1-t) + y1 * t
    dataset['y'] = y
    
    return dataset