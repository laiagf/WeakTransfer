# Active learning 
# Parameters:
#	- n retrain

import pandas as pd
import os
import random
import hydra
import math
import torch
from transfer.config import ROOT_DIR, device

from transfer.core.transferability_experiment import Retrain_experiment
from transfer.core.active_learner import ActiveLearner




@hydra.main(config_path=ROOT_DIR+"experiments/experiment3_al/config", config_name="config.yaml",version_base=None)
def main(cfg):
	

	excluded_source = cfg.source

	exp_type = cfg.exp_type
	

	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	outputdir = hydra_cfg['runtime']['output_dir']

	df = pd.read_csv(os.path.join(ROOT_DIR, 'files/4minDataset.csv'))

	df.Dataset[df.Dataset.isin(['BAL_1','BAL_2','BAL_3'])] = 'BAL'

	experiment = Retrain_experiment(outputdir, exp_type, excluded_source, ROOT_DIR, cfg.budget, cfg.chrono, 
			cfg.tcn_batch_size, cfg.lr, cfg.n_epochs, device=device, random_state=cfg.random_state)

	for s in experiment.retrain_sources: # TODO FIX THHHHHHHHIS
		#experiment = Retrain_experiment(outputdir, exp_type, excluded_source, ROOT_DIR, 0, cfg.chrono, 
		#	cfg.tcn_batch_size, cfg.lr, cfg.n_epochs, device=device)
		al = ActiveLearner(experiment, cfg.initial_sample, cfg.budget, cfg.batch_size, df, s, initial_n_epochs=cfg.initial_n_epochs, n_epochs=cfg.n_epochs, random_state=cfg.random_state, weight_uncertainty=cfg.weighted_al, balance=cfg.balanced_al)
		al.train_model()
		#experiment.save_outputs()

if __name__=='__main__':
	main()


