from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import networkx as nx


def split_seq(seq,size):
    """ Split up seq in pieces of size """
    for i in range(0,len(seq),size):
        if i+size<len(seq) and seq[i+size] - seq[i] == size:
            yield seq[i:i+size]


def gen_data(interval=5,future=7,graphDataPath = 'sparsedDist.csv'):

	DATA_PATH = 'NY/data/'

	# generate table
	raw = pd.read_csv(DATA_PATH+'uber-raw-data-janjune-15.csv',parse_dates=['Pickup_date'])
	raw['Pickup_date']=(raw['Pickup_date'] - np.min(raw['Pickup_date'])).astype(np.int64)//((10e+8)*interval*60)
	grouped = raw.groupby(by=[raw.Pickup_date,raw.locationID])
	x = pd.DataFrame(grouped.size())
	x.reset_index(inplace=True)
	x = x.pivot(index='Pickup_date', columns='locationID',values=0).fillna(0)
	

	#not all 5min intervals between jan and june are included in the database
	missing = set(np.arange(np.max(x.index.values)+1))-set(x.index.values) 
	x.index = x.index.astype(int)

	#create graph
	sparsed = pd.read_csv(DATA_PATH+graphDataPath,index_col=0)
	G = nx.from_numpy_matrix(sparsed.values)
	assert nx.is_connected(G)
	#number of edges
	print('Nodes: {} , Edges: {}'.format(len(G),len(G.edges())))

	#remove useless columns
	drop_cols = list(set(x.columns.values) - set(sparsed.columns.values.astype(int)))
	x = x.drop(drop_cols,axis=1)
	#print(x.columns)

	return x

def gen_seqs(x, ratio = 0.8):


	#Generate sequences to train and test
	seqs = np.array(list(split_seq(x.index.values,19)))
	np.random.shuffle(seqs)
	print('{} total samples'.format(len(seqs)))

	ratio = 0.8
	split = int(len(seqs)*ratio)


	return seqs[:split], seqs[split:]
