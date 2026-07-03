import torch
import warnings
import os
import pandas as pd
import numpy as np
from weakDetector.core.trainers import ClassifierTrainer

#from SpermWhaleDetector.utils.training_loop_tcn import training_loop




class ActiveLearner():
	def __init__(self, experiment, initial_sample, budget, batch_size, df, s, initial_n_epochs=5, n_epochs=1, random_state=0, model=None, weight_uncertainty=False, balance=False):
		"""A class for an experiment that simulates active learning.

		Args:
			experiment (Retrain_experiment): _description_
			budget (integer): Maximum number of recordings that can be audited by the oracle
			batch_size (int): Number of recordings to be audited by the oracle at each step.
			df (pandas dataframe): pandas dataframe of labelled data
			s (str): source of data where the model is being retrained
			n_epochs (int, optional): Number of epochs the model is retrained for at each step. Defaults to 1.
			model (torch model, optional): Model to be retrained if different than model in experiment. Defaults to None.
		"""
		self.experiment = experiment
		self.initial_sample = initial_sample
		self.budget = budget

		self.batch_size=batch_size
		
		self.s = s

		if model is not None:
			self.experiment.model = model.to(self.experiment.device)

		self.df = df[df.Dataset==s].reset_index(drop=True)
		self.df['training']=0
		self.df['embedding_path']=self.experiment._embeddings_path

		self.n_epochs = n_epochs
		self.initial_n_epochs = initial_n_epochs
		self.df_val_epochs = []

		self.random_state = random_state
		self.rng = np.random.RandomState(self.random_state)
		self._weight_uncertainty = weight_uncertainty
		self._balance = balance
		print('active learner initialised with budget', self.budget, 'and batch size', self.batch_size, 'and balanced auditing', self._balance)

	def train_model(self):
		""" Perform active learning training until budget runs out.
		"""
		steps = 0

		optimiser = torch.optim.Adam(self.experiment.model.parameters(), lr=self.experiment.lr)
		self.trainer = ClassifierTrainer(self.experiment.model, optimiser, self.experiment.lr, log_interval=50, lr_decrease_rate=1)

		#Initial training step
		if self.initial_sample>0:
			self.budget = self.budget - self.initial_sample
			audit_indices = self.df.sample(n=self.initial_sample, random_state=self.random_state).index
			self.df.training[self.df.index.isin(audit_indices)] = 1	
			self._training_step(0, n_epochs=self.initial_n_epochs)
			steps+=1

		while self.budget>0:
			print(f'Active learning step {steps}')
			n = min(self.budget, self.batch_size)
			self._simulate_audit_step(n)
			self._training_step(steps, n_epochs=self.n_epochs)  # retrain model with newly annotated data
			#Training step: 
			steps = steps+1
		df_val_data = pd.concat(self.df_val_epochs)
		df_val_data.to_csv(os.path.join(self.experiment.out_dir,f'val_{self.s}_al.csv'), index=False)
		self.df.to_csv(os.path.join(self.experiment.out_dir, f'al_outputs_{self.s}.csv'), index=False)
	
	def _training_step(self, i, n_epochs=1):
		"""Perform a training iteration with available annotations.

		Args:
			i (int): training iteration
		"""

		df_train = self.df[self.df.training==1].reset_index(drop=True)
		df_val = self.df[self.df.training==0].reset_index(drop=True)

		train_loader, val_loader = self.experiment.create_dataloaders(df_train= df_train, df_val=df_val)

		print(f'Training step {i}, training size {len(df_train)}')

		self.trainer(train_loader, val_loader, n_epochs, self.experiment.device)

		self.experiment.model = self.trainer.model.eval()

		retrained_model = self.experiment.model
		
		torch.save(self.experiment.model.state_dict(), os.path.join(self.experiment.out_dir, f'{self.s}_retrained_tcn_{i}.pth'))	
		self.df = self.experiment.iterate_tcn_deployment(model=retrained_model, df = self.df, col_name=f'retrained_outputs_{self.experiment.exp_type}_{self.s}_{i}')
		self.df[f'training_{i}'] = self.df.training
		log_data = self.trainer.training_log

		log_data.to_csv(os.path.join(self.experiment.out_dir,f'val_{self.s}_{i}.csv'), index=False)
		self.df_val_epochs.append(log_data)


		return

	def _score(self):
		"""Compute uncertainty score (entropy) for validation data.

		Returns:
			pandas.DataFrame: dataframe for val data with uncertainty scores
		"""
		model = self.experiment.model
		model.eval()
		df_val = self.df[self.df.training==0]
		df_val['score'] = 0
		df_val['pred_class'] = -1
		for i in df_val.index:
			ys = self.experiment.deploy_tcn(model, i, df_val)
			predicted_class = torch.argmax(ys).item()
			df_val.pred_class[i] = predicted_class
			score =  self._entropy_score(ys).cpu().detach().numpy()
			#print('score', score)
			df_val.score[i] = score

		return df_val


	def _select_audit_indices(self, df, n):

		n = min(len(df), n)
		if self._weight_uncertainty:
			df['probs'] = df.score/df.score.sum()
			audit_indices = self.rng.choice(df.index, size=n, replace=False, p=df.probs)
			print('auditing weighted by uncertainty')
		else:
			df.sort_values('score', ascending=False, inplace=True)
			audit_indices = list(df.index)[:n]
		
		return audit_indices

	def _simulate_audit_step(self, n):
		"""Audit files with highest uncertainty score

		Args:
			n (int): Numbers of files to audit
		"""
		df_val = self._score()
		print('auditing!!!', self._balance, self._weight_uncertainty)
		if len(df_val)==0:
			warnings.warn('ran out of files to audit')
			self.budget=0					

		else:
			n = min(len(df_val), n)
			if self._balance:
				print('auditing balanced')
				df_val_pos = df_val[df_val.pred_class==1]#.sort_values('score', ascending=False)
				df_val_neg = df_val[df_val.pred_class==0]#.sort_values('score', ascending=False)
				
				n_pos = min(len(df_val_pos), n//2)
				n_neg = min(len(df_val_neg), n//2)
				if n_neg<n//2:
					n_pos = n-n_neg
				audit_indices = self._select_audit_indices(df_val_pos, n_pos) + self._select_audit_indices(df_val_neg, n_neg)


			else:
				#df_val.sort_values('score', ascending=False, inplace=True)

#				audit_indices = list(df_val.index)[:n]
				audit_indices = self._select_audit_indices(df_val, n)
			self.budget = self.budget - n
			self.df.training[self.df.index.isin(audit_indices)] = 1

		return

	def _entropy_score(self, ys):
		"""Get entropy uncertainty score from model outputs.

		Args:
			ys (torch.Tensor): Model outputs for each class (after log softmax)

		Returns:
			torch.Float: entropy uncertainty
		"""
		ps = torch.exp(ys)

		return -(ps*torch.log2(ps)).sum()





