import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sys.path.append("../")
from setpts.utils import *
from process.config import *
from setpts.gmm import *
from setpts.bayes import *
from setpts.ornstein_uhlenbeck import *
from setpts.smooth import *

import warnings
warnings.filterwarnings("ignore")


def main(measurements_df, prefix):
    for method in ['bayes', "lowess", "gmm", "ou"]:
        if method == 'gmm':
            setpt_df = run_gmm_model(measurements_df, cluster_picker='healthy')
            setpt_df.to_pickle(f"data/setpts/{prefix}_gmm_healthy.pkl")

            setpt_df2 = run_gmm_model(measurements_df, cluster_picker='thresh')
            setpt_df2.to_pickle(f"data/setpts/{prefix}_gmm_thresh.pkl")
        elif method == 'bayes':
            setpt_df = run_pEB_model(measurements_df, prior_method='reference')
            setpt_df.to_pickle(f"data/setpts/{prefix}_bayes_reference.pkl")

            setpt_df2 = run_pEB_model(measurements_df, prior_method='empirical')
            setpt_df2.to_pickle(f"data/setpts/{prefix}_bayes_empirical.pkl")
        elif method == 'lowess':
            setpt_df = run_lowess_model(measurements_df)
            setpt_df.to_pickle(f"data/setpts/{prefix}_lowess.pkl")
        elif method == "ou":
            setpt_df = run_ou_with_prior(measurements_df)
            setpt_df.to_pickle(f"data/setpts/{prefix}_ou.pkl")
        else:
            raise Exception("Not a valid method!")

if __name__ == "__main__":
    runs = [('../data/filter/labs_filter_cutoff_80_5.pkl', 'cutoff'), 
            ('../data/filter/labs_filter_foy_30_5_2018.pkl', 'foy'), 
            ('../data/filter/labs_filter_yash_30_5_2018.pkl', 'yash')]
    for path, prefix in runs:
        print(f'--------------------{prefix}--------------------')
        main(pd.read_pickle(path), prefix)
    