
import pandas as pd
import os
import random
import hydra
import math
import numpy as np
import torchaudio
import torch
import warnings
warnings.filterwarnings("ignore")

import torchvision.transforms as T

from scipy.signal import butter, lfilter
from joblib import Parallel, delayed



random.seed(0)



class Dj():

	def __init__(self, alpha, beta, files_path, wav_length, out_dir):
		"""Initialise DJ class.

		Args:
			alpha (float): alpha value for beta distribution from where addition proportions are drawn.
			beta (float): beta value for beta distribution from where addition proportions are drawn.
			files_path (string): Path to wavfiles.
			wav_length (int): Expected length of wavfiles after resampling.
			out_dir (string): Directory where to store files.
			feature_engine (): .
		"""
		self.alpha = alpha
		self.beta = beta
		self.files_path = files_path
		self.wav_length = wav_length

		self.out_dir = out_dir

		self.mixup_dir = os.path.join(out_dir, 'synthetic_files')
		if not os.path.exists(self.mixup_dir):
			os.mkdir(self.mixup_dir)
		else:
			print('mixup dir already exists')

		self.feature_engine = None

	

		#self.vae_initialised = False

	def mixup_files(self, f1, f2):
		"""Mix two wavfiles. 

		Args:
			f1 (string): Filename 1.
			f2 (string): Filename 2.

		Returns:
			tuple: tuple containing filename, signal array, and addition of proportion of mixed file.
		"""


		f = 'mixup_'+f1[:-4]+'_'+f2


		w1 = self.feature_engine.load_file(os.path.join(self.files_path, f1))
		w2 = self.feature_engine.load_file(os.path.join(self.files_path, f2))

		real_spec1 = self.feature_engine._tf(w1[0, :])
		real_spec2 = self.feature_engine._tf(w2[0, :])	

		
		med1 = np.median(abs(real_spec1).sum(axis=0))
		med2 = np.median(abs(real_spec2).sum(axis=0))
		w1 = w1*(med2/med1)

		w1 = self._right_pad_if_necessary(w1)
		w2 = self._right_pad_if_necessary(w2)


		#Add!
		b = np.random.beta(self.alpha, self.beta)
		w = (1-b)*w1+(b)*w2

		return f, w, b

	def extract_mixup_features(self, f1, f2, mixup_function=None, window_length=2048):
		"""Create feature sequence of the mix between two files.
		Args:
			f1 (string): Filename 1.
			f2 (string): Filename 2.
			mixup_function (_type_, optional): _description_. Defaults to None.
			window_length (int, optional): _description_. Defaults to 2048.

		Returns:
			_type_: _description_
		"""
		if mixup_function is None:
			mixup_function = self.mixup_files

		f, w, b = mixup_function(f1, f2)

		torchaudio.save(os.path.join(self.mixup_dir, f), w.float(), 48000)

		seq = self.feature_engine.extract(w)
		
		torch.save(torch.clone(seq.cpu()), os.path.join(self.mixup_dir, f[:-4]+'.pt'))

		return f, b

	def _right_pad_if_necessary(self, w):
	
		signal_length = w.shape[-1]
		if signal_length<self.wav_length:
			last_dim_padding = (0, self.wav_length-signal_length)
			w = torch.nn.functional.pad(w, last_dim_padding)

		return w

	
	def _cut_if_necessary(self, w, dim=1):
		"""Cuts signal to class' target_length along dimension dim.

		Args:
			w (torch tensor): waveform signal to cut.
			dim (int, optional): Dimension to cut along. Defaults to 1.

		Raises:
			ValueError: Specified dimension does not exist in signal

		Returns:
			torch tensor: Cut signal, or original signal if cutting was not needed. 
		"""

		if dim>=len(w.shape):
			raise ValueError('Specified dimension is larger than number of dimensions of signal.')

		if w.shape[dim] > self.wav_length:		
			slicing = [slice(None)] * w.ndimension()
			slicing[dim] = slice(0, self.wav_length)
			w = w[tuple(slicing)]

		return w

class Dj_CT(Dj):
	def __init__(self, alpha, beta, files_path, wav_length, out_dir, ct_path, min_snr=5,  mu=1, sigma=1, min_seconds_overlap=30, sr=48000, max_snr=25):
		"""Initialize ClickTrain DJ (CT_DJ).

		Args:
			alpha (_type_): _description_
			beta (_type_): _description_
			files_path (_type_): _description_
			wav_length (_type_): _description_
			out_dir (_type_): _description_
			ct_path (_type_): _description_
			min_snr (int, optional): _description_. Defaults to 5.
			mu (int, optional): _description_. Defaults to 1.
			sigma (int, optional): _description_. Defaults to 1.
			min_seconds_overlap (int, optional): _description_. Defaults to 30.
			sr (int, optional): _description_. Defaults to 48000.
			max_snr (int, optional): _description_. Defaults to 25.
		"""
		super().__init__(alpha, beta, files_path, wav_length, out_dir)
		self.ct_path = ct_path
		self.min_snr = min_snr
		self.mu = mu
		self.sigma = sigma
		self.min_overlap = min_seconds_overlap*sr
		self.max_snr = max_snr

	def prepare_ct_2_overlap(self, w, n=None):
		"""Prepare click train to be added to noise wavfile.

		Args:
			w (_type_): _description_
			n (_type_, optional): _description_. Defaults to None.

		Returns:
			_type_: _description_
		"""
		if n is None:
			n = self.wav_length

		#Find n_channels
		n_ch = w.shape[0]

		# Find start index
		s2 = np.random.randint(self.min_overlap-w.shape[1], n-self.min_overlap)

		start_ind = max(-s2, 0)
		end_ind = min(start_ind+n, w.shape[1])

		w_overlap = w[:, start_ind:end_ind]


		left_pad = torch.zeros(n_ch, max(0, s2))

		right_pad =torch.zeros(n_ch, max(0, n-w_overlap.shape[1]-left_pad.shape[1]))


		w_overlap = torch.nan_to_num(torch.cat((left_pad, w_overlap, right_pad), 1)[:, :n])
		
			#print(f'n:{n}\n s2:{s2}\n left_pad:{left_pad.shape}\n w_overlap: {w_overlap.shape}\n right_pad: {right_pad.shape}\n')

		return start_ind, end_ind, w_overlap


	def add_ct_2_file(self, f1, f2):
		"""Add clicktrain to wavfile.

		Args:
			f1 (_type_): _description_
			f2 (_type_): _description_
		"""

		snr = min(self.min_snr+np.random.lognormal(self.mu, self.sigma), self.max_snr)
		
		f = 'synthetictrain_'+f1[:-4]+'_'+f2

		df_clicks = pd.read_csv(os.path.join(self.ct_path, f2[:-3]+'csv'))

		w1, sr1 = torchaudio.load(os.path.join(self.files_path,f1))
		w2, sr2 = torchaudio.load(os.path.join(self.ct_path,f2))


		w1 = self._cut_if_necessary(w1)

		start_ind, end_ind, w_overlap = self.prepare_ct_2_overlap(w2, n=w1.shape[1])

		df_overlapping_clicks = df_clicks[(df_clicks.resampled_start_sample>=start_ind) & (df_clicks.resampled_end_sample<=end_ind+1)]

		#choose_channel 
		rms_ch1 = df_overlapping_clicks.rms1.median()
		rms_ch2 = df_overlapping_clicks.rms2.median()
		if rms_ch1>rms_ch2:
			w_overlap = w_overlap[:1, :]
			rms_clicks = rms_ch1
		else:
			w_overlap = w_overlap[1:, :] 
			rms_clicks=rms_ch2

		rms_file = self._rms(w1[0, :])

		k = (rms_file/rms_clicks) * (10**(snr/10))
		w = w1+k*w_overlap

		if torch.isnan(w).any():
			print('Nan before bp')
			print(len(df_overlapping_clicks))
			print(f2, start_ind, end_ind)
			#print(torch.isnan(w1).any(), torch.isnan(w_overlap).any())
			#print('k', k, 'rms_file', rms_file, 'rms_clicks', rms_clicks, 'snr', snr)


		w = self._bandpass(w, lowcut=1000, highcut=20000, fs=48000)
		## DEBUG
		if torch.isnan(w).any():
			print('Nan after bp')
	#	torchaudio.save(os.path.join(self.out_dir, f+'.wav'), w.float(), 48000)

		return f, w, (snr, start_ind, end_ind)


	#TODO get this from somewhere else
	def _rms(self, signal):
		# Auxiliary function to compute root mean square (rms) of sequence (in this case, waveform)
		if len(signal.shape)==1:
			scalar_prod = torch.dot(signal[:], signal[:])
		else: 
			scalar_prod = torch.dot(signal[0, :], signal[0, :])
		rms = math.sqrt(scalar_prod/len(signal))
		return rms
	def _bandpass(self, signal, fs, lowcut, highcut, order=6):
		# Auxiliary function to apply butterworth bandpass filter to waveform.
		# Input: waveform, sampling rate, lower frequency, higher frequency, order of filter (default = 5)
		b,a = butter(order, [lowcut, highcut], fs=fs, btype='band')
		filtered_signal = torch.tensor(lfilter(b, a, signal))
		return filtered_signal


class Synthetic_mixer():
	def __init__(self, dj, df_neg,  n_retrain, output_type, 
		df_pos=None, x_aug=1, random_state=0):
		"""Initialize synthetic mixer

		Args:
			dj (_type_): _description_
			df_neg (_type_): _description_
			n_retrain (_type_): _description_
			output_type (_type_): _description_
			df_pos (_type_, optional): _description_. Defaults to None.
			x_aug (int, optional): _description_. Defaults to 1.
		"""

		self.dj = dj
		self.df_neg = df_neg 
		self.x_aug = x_aug 
		self.n_retrain = n_retrain
		self.n_class = int(x_aug*n_retrain//2)
		if not df_pos is None:
			df_pos = df_pos.sample(frac=1, random_state=random_state).reset_index(drop=True)
			self.df_pos = df_pos
		self.output_type = output_type
		self.random_state = random_state
		self.mix = self.dj.extract_mixup_features


	def create_synthetic_positives(self):
		"""Create synthetic positive files using noise files from new site and positive files from other sites.

		Returns:
			_type_: _description_
		"""

		#self.sample_pos = self.df_pos.sample(self.n_class, replace=True, random_state=1).reset_index(drop=True)
		sample_pos = self.df_pos[:self.n_class]
		pos_files = list(sample_pos.FileName)
		neg_files = list(self.df_neg.FileName)*self.x_aug
		random.Random(self.random_state).shuffle(neg_files)

		fs = []
		bs = []
		for i in range(self.n_class):
			f, b = self.mix(pos_files[i], neg_files[i])
			fs.append(f),
			bs.append(b)


		print(f'Lengths of files {len(pos_files)}, {len(neg_files)}, {len(list(self.df_neg.FileName))}')
		self.df_mixup_pos = pd.DataFrame({'File1': pos_files, 'File2': neg_files, 
			'FileName':fs, 'files_dir':self.dj.mixup_dir, 'b':bs, 'Label':1})

		return self.df_mixup_pos


	def create_synthetic_negatives(self):
		"""Create dataframe of background (negative) files.

		Returns:
			_type_: _description_
		"""
		f1s = []
		f2s = []
		fs = []
		bs = []
		for i in range(self.x_aug-1):
			for j in self.df_neg.index:
				f1 = self.df_neg.FileName[j]
				f2 = random.choice(list(self.df_neg.FileName[self.df_neg.FileName!=f1]))

				f1s.append(f1)
				f2s.append(f2)
				f, b = self.mix(f1, f2)	
				fs.append(f),
				bs.append(b)		

		self.df_mixup_neg = pd.DataFrame({'File1': f1s, 'File2': f2s, 'FileName':fs, 
			'files_dir':self.dj.mixup_dir, 'b':bs, 'Label':0})


		return self.df_mixup_neg


	def create_synthetic_train(self):
		"""Create dataframe of synthetic files to train the model

		Returns:
			_type_: _description_
		"""

		df_mixup_pos = self.create_synthetic_positives()

		df_mixup_neg = self.create_synthetic_negatives()



		df_mixup = pd.concat([df_mixup_pos, df_mixup_neg]).reset_index(drop=True)
		print(self.df_neg.columns)
		df_train = pd.concat([df_mixup, self.df_neg[['FileName', 'Label', 'files_dir']]]).reset_index(drop=True)
		df_train = df_train.sample(frac=1).reset_index(drop=True)
		

		return df_train


class Synthetic_CT_mixer(Synthetic_mixer):
	def __init__(self, dj, df_neg, df_cts, n_retrain, output_type, random_state=0):
		super().__init__(dj, df_neg, n_retrain, output_type, random_state=random_state)
		
		df_cts = df_cts.sample(frac=1, random_state=random_state).reset_index(drop=True)

		self.df_cts = df_cts


	def create_synthetic_positives(self):
		"""Create synthetic positive files using noise files from new site and clicktrain files.

		Returns:
			_type_: _description_
		"""
		fs = []
		f1s = []
		#cts = []
		snrs = []
		start_inds = []
		end_inds = []
		self.cts_sample = self.df_cts[:self.n_class]# self.df_cts.sample(self.n_class, replace=True, random_state=1).reset_index(drop=True)
		pos_files = list(self.cts_sample.ClickTrain_Name)
		
		neg_files = list(self.df_neg.FileName)*self.x_aug
		random.shuffle(neg_files)

		
		"""
		for i in range(self.n_class):
			# TODO check that pos and neg files are always in the same order
			f, (snr, start_ind, end_ind) = self.mix(neg_files[i], pos_files[i], self.dj.add_ct_2_file)
			# TODO store start of click train.... this needs rethinking
			fs.append(f)
			f2s = neg_files[i]
			f1s = pos_files[i]
			snrs.append(snr)
			start_inds.append(start_ind)
			end_inds.append(end_ind)
		
		"""
		results = Parallel(n_jobs=8)(delayed(self.mix)(neg_files[i], pos_files[i], self.dj.add_ct_2_file) for i in range(self.n_class))
		
		fs, details = zip(*results)
		snrs, start_inds, end_inds = zip(*details)
		f2s = [neg_files[i] for i in range(self.n_class)]
		f1s = [pos_files[i] for i in range(self.n_class)]	
		
		self.df_mixup_pos = pd.DataFrame({'File1': f1s, 'File2': f2s, 'FileName':fs, 
			'files_dir':os.path.join(self.dj.out_dir, 'synthetic_files/'), 'snr':snrs, 'start_ind':start_inds, 'end_ind':end_inds, 'Label':1})

		return self.df_mixup_pos





