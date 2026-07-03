# Retrains on new site
# Parameters:
# 	- n_files for retrain
#	- order: random, chronological

import hydra
from transfer.config import ROOT_DIR, device   

from transfer.core.transferability_experiment import Retrain_experiment



@hydra.main(config_path=ROOT_DIR+"experiments/experiment1_retrain/config", config_name="config.yaml",version_base=None)
def main(cfg):
	

	excluded_source = cfg.source
	exp_type = cfg.exp_type


	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	outputdir = hydra_cfg['runtime']['output_dir']
	

	experiment = Retrain_experiment(outputdir, exp_type, excluded_source, ROOT_DIR, cfg.n_retrain, cfg.chrono, 
		cfg.batch_size, cfg.lr, cfg.n_epochs, device=device,freeze_tcn=cfg.freeze_tcn, random_state=cfg.random_state, min_snr=cfg.min_snr)

	for s in experiment.retrain_sources:

		experiment.retrain_tcn(s)


	experiment.save_outputs()

if __name__=='__main__':
	main()