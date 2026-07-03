import torch
import math
import os

def deploy_tcn(swDataset, model, device, col_name='output'):



	model.eval()
	outputs = []

	df = swDataset.annotations


	for j in df.index:

		t = swDataset.load_item(df.FileName[j][:-3]+'pt')

		t = t.resize(1, t.shape[0], t.shape[1])
		with torch.no_grad():
			output = model(t.to(device).float())
			p = math.exp(output.cpu().detach().numpy()[0][1])
			outputs.append(p)

	df[col_name] = outputs
	return df
