from weakDetector.config import SOURCES, CODE_DIR
#from transfer.config import TRANSF_SOURCES
import os

def get_run_dir(input_type, val_source, latent_size=32, min_snr=0, target_seconds=240, random_state=9):
    #print('random state in get_run_dir', random_state)
    if val_source=='BAL':
        train_sources = [s for s in SOURCES if not s in ['BAL_1', 'BAL_2', 'BAL_3']]
    else:
        train_sources = [s for s in SOURCES if s != val_source]

    train_sources.sort()
    if input_type == 'RMS':
        rundir = CODE_DIR + f"experiments/experiment_HandcraftedFeatures/run_outputs/{target_seconds}/features=RMS/split=by_source,train_sources={train_sources},min_snr={min_snr}/HR_5/{random_state}/"
    elif input_type == 'VAE':
        rundir = CODE_DIR + f"experiments/experiment_VAE/run_outputs/{target_seconds}/dataset=spectrogram/split=by_source,train_sources={train_sources},min_snr={min_snr}/{latent_size}/random_state={random_state}/"
    else:
        raise ValueError (f'Invalid input type {input_type}. Choose either "RMS" or "VAE".')
    if not os.path.exists(rundir+'trained_tcn.pth'):
        raise ValueError(f"Trained TCN model not found in {rundir}")

    print('Original run directory:', rundir)
    return rundir