import torch


ROOT_DIR = '/home/laia/Projects/transferability/'

WAV_PATH = '/mnt/spinning1/processedAudioFiles/' 


#SOURCES = ['BAL_1','BAL_2','BAL_3','CAL','AS','ICE','MED', 'CS']

TRANSF_SOURCES = ['CS', 'ICE', 'MED']

source_dict =  {'BAL':['BAL_1', 'BAL_2', 'BAL_3'], 'CAL':['CAL'], 'MED':['MED'], 'CS':['CS'], 'AS':['AS'], 'ICE':['ICE']}

WD_DIR = '/home/laia/Projects/WeakDetector/'

CT_PATH = '/media/laia/Backup2/synthetic_waves/'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
