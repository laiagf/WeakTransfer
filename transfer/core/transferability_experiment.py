import pandas as pd 
import os
import torch 
from torch.utils.data import DataLoader
import math
import random
import copy
from omegaconf import OmegaConf
from weakDetector.models.tcn import TCN
from weakDetector.models.vae_resnet import VAE_ResNet
from weakDetector.datasets.spermWhaleDataset import SpermWhaleDataset
from weakDetector.core.trainers import ClassifierTrainer
from weakDetector.core.featureEngines import VAEFeatureExtractor, HeuristicFeatureExtractor
from weakDetector.config import DATA_PATH
from weakDetector.config import ROOT_DIR as WD_ROOT_DIR
from weakDetector.config import CODE_DIR as WD_DIR
from weakDetector.utils.mm import get_target_length

from transfer.utils.exp_utils import get_run_dir

from transfer.config import ROOT_DIR
#from transfer.order.datasets.stdsedTensorEmbeddingDataset import stdsedTensorEmbeddingDataset
#from transfer.datasets.stdsedTensorEmbeddingDataset import stdsedTensorEmbeddingDataset
from transfer.core.dj import Synthetic_mixer, Synthetic_CT_mixer

class Transferability_experiment():

	def __init__(self, out_dir, exp_type, excluded_source, ROOT_DIR, random_state, device='cuda', df_sw_col='chp1_tcn_dir', min_snr=0 ):

		self.source_dict = {'BAL_1':['BAL'], 'BAL_2':['BAL'], 'BAL_3':['BAL'], 'CAL':['CAL'], 'MED':['MED'], 'CS':['CS'], 'AS':['AS'], 'ICE':['ICE'], 'BAL':['BAL']}

		self.excluded_source = excluded_source

		#df_sw_runs = df_sw_runs[(df_sw_runs.excluded_sources==excluded_source) & (df_sw_runs.exp_type==exp_type)].reset_index(drop=True)

		self.random_state = random_state
		self.df_sw_col = df_sw_col

		self.out_dir = out_dir

		self.exp_type = exp_type
		rundir = get_run_dir(exp_type, excluded_source, random_state=random_state)

		if exp_type == 'RMS':
			self._set_attributes_rms(rundir)
		elif exp_type == 'VAE':
			self._set_attributes_vae(rundir)

		self.min_snr = min_snr
		self.df = pd.read_csv(WD_DIR+'files/4minDataset.csv')
		self.df.Dataset[self.df.Dataset.isin(['BAL_1','BAL_2','BAL_3'])] = 'BAL'

		#if df_sw_col=='chp1_tcn_dir_minsnr6':
		#	self.df = self.df[self.df.SNR_999>=6].reset_index(drop=True)
		self.df = self.df[self.df.SNR_999>=min_snr].reset_index(drop=True)
		
		
		self._swDataset.annotations = self.df

		self.device = device

		self.ROOT_DIR = ROOT_DIR

	def _set_attributes_rms(self, tcn_path):

		self.tcn_path = tcn_path
		cfg_tcn = OmegaConf.load(os.path.join(self.tcn_path, '.hydra/config.yaml'))
		#cfg_tcn['model'] = cfg_tcn.parameters #config files are a bit different

		self.cfg_tcn = cfg_tcn
		self._embeddings_path = os.path.join(DATA_PATH, 'RMS_Vectors/240/RMS_HR_5bands/')
		self.latent_size = cfg_tcn.n_channels
		self.df_standard=None
		self.normalise=True
		#self.seq_length = int(48000*60*4//512)
		self._target_length = int(48000*60*4//512)

		self._swDataset = SpermWhaleDataset(annotations_file=cfg_tcn.annotations_file, 
											files_dir=self._embeddings_path, 
											target_length=self._target_length)

	def _set_attributes_vae(self, tcn_path):	
		cfg_tcn = OmegaConf.load(os.path.join(tcn_path, '.hydra/config.yaml'))
		self.cfg_tcn = cfg_tcn
		#ae_path = os.path.join(WD_ROOT_DIR+'experiments/experiment_VAE/train_vae/run_outputs/')+cfg_tcn['run_path']
		vae_run_path = os.path.join(WD_ROOT_DIR, f'experiments/experiment_VAE/train_vae/run_outputs/dataset={cfg_tcn.dataset}/{cfg_tcn.split}_split,sources={cfg_tcn.train_sources}/{cfg_tcn.latent_size}/random_state=2/') 	
		# TODO - put this ina  function
		
		
		self._embeddings_path = os.path.join(vae_run_path, 'embeddings/240/')	

		self.ae_path = vae_run_path
		self.tcn_path = tcn_path
		self.df_standard = pd.read_csv(os.path.join(vae_run_path, 'standard_dict.csv'))

		self._target_length = get_target_length(cfg_tcn)

		cfg_ae = OmegaConf.load(os.path.join(vae_run_path , '.hydra/config.yaml'))

		self.latent_size = cfg_ae.model.latent_size

		#self.seq_length = int(48000*60*4//(128*256))

		self._swDataset = SpermWhaleDataset(annotations_file=cfg_tcn.annotations_file, 
											files_dir=self._embeddings_path, 
											target_length=self._target_length, channels=[i for i in range(self.latent_size)])

	def load_pretrained_tcn(self, softmax=True, load_weights=True):
		self.model = TCN(self.latent_size, self.cfg_tcn.model.output_size, [self.cfg_tcn.model.n_hid]*self.cfg_tcn.model.levels, 
					kernel_size=self.cfg_tcn.model.kernel_size, dropout=self.cfg_tcn.model.dropout)


		if load_weights:
			self.model.load_state_dict(torch.load(os.path.join(self.tcn_path,'trained_tcn.pth')))
		self.model.to(self.device)

		self.model.eval()

		return self.model

	def iterate_tcn_deployment(self, df=None, col_name=None, model=None, override_df=True):

		
		if df is None:
			df = self.df
		if model is None:
			model = self.model
		if col_name is None:
			col_name=f'outputs_exc_{self.exp_type}_{self.excluded_source}'


		model.eval()
		outputs = []
		for j in df.index:
			y = self.deploy_tcn(model, j)
			p = math.exp(y.cpu().detach().numpy()[0][1])
			outputs.append(p)

		df[col_name] = outputs

		if override_df:
			self.df = df

		return df

	def deploy_tcn(self, model, i, df_val=None):
		
		model.eval()

		if not df_val is None:
			aux_Dataset = self._swDataset
			aux_Dataset.annotations = df_val
			t = aux_Dataset.load_item(self.df.FileName[i][:-3]+'pt')
		else:
			t = self._swDataset.load_item(self.df.FileName[i][:-3]+'pt', self._embeddings_path)

		t = t.resize(1, t.shape[0], t.shape[1])
		with torch.no_grad():
			output = model(t.to(self.device).float())
		
			
		return output
	
	def find_metrics(self, df=None, col_name=None):
		if df is None:
			df = self.df
		if col_name is None:
			col_name=f'outputs_exc_{self.exp_type}_{self.excluded_source}'

		tp = len(df[(df[col_name]>=0.5) & (df.Label==1)])
		tn = len(df[(df[col_name]<0.5) & (df.Label==0)])
		fp = len(df[(df[col_name]>=0.5) & (df.Label==0)])
		fn = len(df[(df[col_name]<0.5) & (df.Label==1)])
		print(tn, tp, fn, fp)
		acc = (tp+tn)/(tp+tn+fp+fn)
		recall = tp/(tp+fn)
		tnr = tn/(tn+fp)

		return acc, recall, tnr

	def save_outputs(self, output_name = 'outputs.csv'):

		self.df.to_csv(os.path.join(self.out_dir, output_name), index=False)





class AdaBN_experiment(Transferability_experiment):
	def __init__(self, out_dir, exp_type, excluded_source, ROOT_DIR, random_state, device='cuda'):
		super().__init__(out_dir, exp_type, excluded_source, ROOT_DIR, device, random_state=random_state, df_sw_col='chp1_tcn_dir_adabn')

		self.load_pretrained_tcn()

		self._target_df = self.df[self.df.Dataset==excluded_source].reset_index(drop=True)
		self._target_swDataset = copy.copy(self._swDataset)
		self._target_swDataset.annotations = self._target_df

	def _compute_neuron_activations(self):

		all_activations = []
		

		for i in self._target_df.index:
			t = self._target_swDataset.load_item(self._target_df.FileName[i][:-3]+'pt')
			t = t.resize(1, t.shape[0], t.shape[1])
			with torch.no_grad():
				_, neuron_act = self.model(t.to(self.device).float(), log_weights=True)
			all_activations.append(neuron_act)
		return all_activations
	
	def update_bn(self):
		n_blocks = len(self.model.tcn.layers)

		for n_block in range(n_blocks):
			#CONV1
			i=2*n_block
			#print(i)
			all_activations = self._compute_neuron_activations()
			layer_activations = []
			for j in range(len(all_activations)):
				layer_activations.append(all_activations[j][i])
			layer_activations = torch.stack(layer_activations)

			mu = torch.mean(layer_activations, axis=[0,1,3])
			var = torch.var(layer_activations, axis=[0,1,3])
			self.model.tcn.layers[n_block].batch_norm1.running_mean = mu
			self.model.tcn.layers[n_block].batch_norm1.running_var = var

			#CONV2
			i=2*n_block+1
			#print(i)
			all_activations = self._compute_neuron_activations()
			layer_activations = []
			for j in range(len(all_activations)):
				layer_activations.append(all_activations[j][i])
			layer_activations = torch.stack(layer_activations)

			mu = torch.mean(layer_activations, axis=[0,1,3])
			var = torch.var(layer_activations, axis=[0,1,3])
			self.model.tcn.layers[n_block].batch_norm2.running_mean = mu
			self.model.tcn.layers[n_block].batch_norm2.running_var = var


class Retrain_experiment(Transferability_experiment):
	def __init__ (self, out_dir, exp_type, excluded_source, ROOT_DIR, n_retrain, chrono, batch_size, lr, 
		n_epochs, device='cuda', freeze_tcn=False, load_weights=True, random_state=0, min_snr=0):
		print('Random state', random_state)
		super().__init__(out_dir, exp_type, excluded_source, ROOT_DIR, random_state=random_state,  device=device, min_snr=min_snr)
		
		self.load_pretrained_tcn(load_weights=load_weights)

		if freeze_tcn:
			self.model.tcn.requires_grad_(False)

		self.n_retrain = n_retrain

		self.chrono = chrono

		self.batch_size = batch_size

		self.lr = lr

		self.n_epochs = n_epochs

		self.retrain_sources = self.source_dict[self.excluded_source]

		self.freeze_tcn = freeze_tcn

	def create_chrono_dfs(self, s):

		df_s = self.df[self.df.Dataset==s].reset_index(drop=True)
		df_s['embedding_path'] = self.embedding_path
		df_s.datetime = pd.to_datetime(df_s.datetime)
		df_s.sort_values('datetime').reset_index(drop=True)
		
		pos_index = list(df_s.index[(df_s.Label==1)][0:int(self.n_retrain//2)])
		
		neg_index=list(df_s.index[(df_s.Label==0)][0:int(self.n_retrain//2)])
		
		train_index = pos_index+neg_index
		
		df_train = df_s[df_s.index.isin(train_index)].reset_index(drop=True)
		#anything after the last train value will be test data
		max_train_index = max(train_index)
		df_val = df_s[df_s.index>max_train_index].reset_index(drop=True)
		return df_train, df_val

	def create_random_dfs(self, s):

		df_s = self.df[self.df.Dataset==s].reset_index(drop=True)
		df_s['embedding_path'] = self._embeddings_path

		#pos_index = random.sample(list(df_s.index[(df_s.Label==1)]), int(self.n_retrain//2))
		#neg_index = random.sample(list(df_s.index[(df_s.Label==0)]), int(self.n_retrain//2))
		#train_index = pos_index+neg_index
		df_pos = df_s[df_s.Label==1]
		df_neg = df_s[df_s.Label==0]
		df_pos = df_pos.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
		df_neg = df_neg.sample(frac=1, random_state=self.random_state).reset_index(drop=True)


		df_train = pd.concat([df_pos[:int(self.n_retrain//2)], df_neg[:int(self.n_retrain//2)]]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
		df_val = pd.concat([df_pos[int(self.n_retrain//2):], df_neg[int(self.n_retrain//2):]]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

		#df_train = df_s[df_s.index.isin(train_index)].reset_index(drop=True)
		

		#df_val = df_s[~df_s.index.isin(train_index)].reset_index(drop=True)
		return df_train, df_val

	def create_dataloaders(self, s=None, df_train=None, df_val=None):
		
		if (df_train is None) or (df_val is None):
			if self.chrono:
				df_train, df_val = self.create_chrono_dfs(s)
			else:
				df_train, df_val = self.create_random_dfs(s)

			df_train.to_csv(os.path.join(self.out_dir, f'retrain_df_{s}.csv'), index=False)


		train_set = SpermWhaleDataset(df_train, self._embeddings_path, 	target_length=self._target_length, 
								sources='all', min_snr=self.min_snr, df_standard=self.df_standard,
								channels=[i for i in range(self.latent_size)])
		val_set = SpermWhaleDataset(df_val, self._embeddings_path, 	target_length=self._target_length, 
								sources='all', min_snr=self.min_snr, df_standard=self.df_standard,
								channels=[i for i in range(self.latent_size)])
		#train_set = stdsedTensorEmbeddingDataset(df_train, self.seq_length)
		#val_set = stdsedTensorEmbeddingDataset(df_val, self.seq_length)

		print(f'len train_set: {len(train_set)}, batch_size: {self.batch_size}')

		train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
		val_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True)

		return train_loader, val_loader

	def retrain_tcn (self, s):

		train_loader, val_loader = self.create_dataloaders(s)

		optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.model.train()

		trainer = ClassifierTrainer(self.model, optimiser, self.lr, log_interval=300)


		trainer(train_loader, val_loader, self.n_epochs, self.device)
		#retrained_model, df_val_data = training_loop(train_loader, val_loader, self.n_epochs, self.model, self.device, 
		#	self.seq_length, optimiser, self.latent_size, self.batch_size, self.lr, 5)

		retrained_model = trainer.model
		log_data = trainer.training_log

		self.model.eval()

		# save retrained model
		torch.save(retrained_model.state_dict(), os.path.join(self.out_dir, f'{s}_retrained_tcn.pth'))	


		self.iterate_tcn_deployment(model=retrained_model, col_name=f'retrained_outputs_{self.exp_type}_{s}')

		log_data.to_csv(os.path.join(self.out_dir,f'val_{s}.csv'), index=False)





class Synthetic_retrain_experiment(Retrain_experiment):

	def __init__(self, out_dir, exp_type, excluded_source, ROOT_DIR, n_retrain, chrono, 
		batch_size, lr, n_epochs, dj, aug_type, random_state, x_aug=1, device='cuda', min_snr = 0, min_snr_synthetic=0):
		
		super().__init__(out_dir, exp_type, excluded_source, ROOT_DIR, n_retrain, chrono, 
			batch_size, lr, n_epochs, device=device, random_state=random_state, min_snr=min_snr)
		
		# TODO add check that n_retrain is an even number
		self.dj = dj

		self.min_snr_synthetic = min_snr_synthetic
		
		self.aug_type = aug_type
		
		self.x_aug = x_aug

		

		self._initialise_feature_engine()


	def _initialise_feature_engine(self):
		print('initialising feature engine')
		if self.exp_type == 'VAE':
			encoded_dim = int(self.latent_size)
			ae_model = VAE_ResNet(encoded_dim).to(self.device)
			ae_model.load_state_dict(torch.load(os.path.join(self.ae_path,'trained_vae.pth'), map_location=self.device))
		
			self.dj.feature_engine = VAEFeatureExtractor(window_size=256*128,  latent_size=encoded_dim, 
			input_type='spectrogram', model=ae_model,  target_length=4*60*48000, sampling_rate=48000, device=self.device)		
		elif self.exp_type == 'RMS':
			self.dj.feature_engine = HeuristicFeatureExtractor(window_size=2048, target_length=4*60*48000, sampling_rate=48000, bp_freqs=[1000, 20000], bp_order=6,
			  rms = True, rms_freqs = [1000, 2000, 4000, 8000, 16000,20000])	
		else:
			print('error')


	def _create_neg_df(self, s):
		if self.chrono:
			df_s = self.df[self.df.Dataset==s].reset_index(drop=True)
			df_s['files_dir'] = self.embedding_path
			df_s.datetime = pd.to_datetime(df_s.datetime)
			df_s.sort_values('datetime').reset_index(drop=True)
			neg_index=list(df_s.index[(df_s.Label==0)][0:int(self.n_retrain//2)])
			df_neg = df_s[df_s.index.isin(neg_index)].reset_index(drop=True)
			return df_neg
		else:
			df_s = self.df[self.df.Dataset==s].reset_index(drop=True)
			df_s['files_dir'] = self._embeddings_path

			df_neg = df_s[df_s.Label==0].reset_index(drop=True)
			df_neg = df_neg.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

			df_neg = df_neg[:int(self.n_retrain//2)]
			return df_neg

	def _initialize_mixer(self, s):

		df_neg = self._create_neg_df(s)

		if self.aug_type=='file':
			df_extra = self.df[(~self.df.Dataset.isin(self.retrain_sources)) & (self.df.Label==1) & (self.df.SNR_999>=self.min_snr_synthetic)].reset_index(drop=True)
			sm = Synthetic_mixer(self.dj,df_neg, self.n_retrain, self.exp_type, df_pos=df_extra, x_aug=self.x_aug)
			return sm

		elif self.aug_type=='ct':
			df_cts = pd.read_csv(os.path.join(self.ROOT_DIR, 'files/clicktrains.csv'))
			sm = Synthetic_CT_mixer(self.dj, df_neg, df_cts, self.n_retrain, self.exp_type)

			return sm
		else: 
			raise Exception (f'Unknown value for aug_type: {self.aug_type}')


	def create_dataloaders(self, s):

		sm = self._initialize_mixer(s)

		df_train =  sm.create_synthetic_train()
		df_train.to_csv(os.path.join(self.out_dir, f'retrain_df_{s}.csv'), index=False)
		df_train['Dataset'] = self.excluded_source
		df_val = self.df[(~self.df.FileName.isin(list(df_train.FileName))) & (self.df.Dataset.isin(self.source_dict[self.excluded_source]))].reset_index(drop=True)
		df_val['embedding_path'] = self._embeddings_path


		train_set = SpermWhaleDataset(df_train, 'variable', target_length=self._target_length, 
								sources='all', min_snr=self.min_snr, df_standard=self.df_standard,
								channels=[i for i in range(self.latent_size)])
		val_set = SpermWhaleDataset(df_val, self._embeddings_path, 	target_length=self._target_length, 
								sources='all', min_snr=self.min_snr, df_standard=self.df_standard,
								channels=[i for i in range(self.latent_size)])


#		train_set = stdsedTensorEmbeddingDataset(df_train, self.seq_length)
#		val_set = stdsedTensorEmbeddingDataset(df_val, self.seq_length)
		print(f'Train set size {len(train_set)}, Val set size {len(val_set)}')

		train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
		val_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True)

		return train_loader, val_loader






