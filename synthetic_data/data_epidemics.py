import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.SISModel as sis
import networkx as nx
import random
import numpy as np

def simulate_SIS(G, timesteps, beta, lambda_, initial_nodes): ##GVA index
    model = sis(G)
    
    cfg = mc.Configuration()
    infected_nodes = initial_nodes
    cfg.add_model_initial_configuration("Infected", infected_nodes)
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('lambda', lambda_)
    
    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(timesteps)
    
    trends = model.build_trends(iterations)
    
    return trends, iterations, model

#Not fully optimized but takes acceptable amount of time to run
def generateInfectionMatrix(G,timesteps, beta, lambda_, initial_nodes):
    
    trends, iterations, modelInfected = simulate_SIS(G,timesteps, beta, lambda_, initial_nodes)
    
    
    tmp = np.array(list(iterations[0]['status'].values()))
  
    mat = np.zeros((len(iterations),tmp.shape[0]))
    for i in range(len(iterations)):    
        for key in iterations[i]['status']:
            tmp[key]=iterations[i]['status'][key]

        mat[i,:] = tmp
        
    return mat.transpose()

def genTrainData(g_nx,beta,sbeta,lambda_,slambda, initial_nodes, batch_size, timesteps,label):   
    
    n_nodes = len(g_nx)
    X = np.zeros([batch_size, n_nodes, timesteps], np.int32)

    betaVec   = np.abs(np.random.normal(beta, beta/10, batch_size))
    lambdaVec = np.abs(np.random.normal(lambda_, lambda_/10, batch_size))

    Y = zip(betaVec,lambdaVec)

    for i,(b,l)in enumerate(Y):
        X[i] = generateInfectionMatrix(g_nx,timesteps = timesteps,  beta=b, lambda_=l, initial_nodes = initial_nodes)

    #Y = np.array(list(Y))   
    Y = (np.ones((batch_size,1))*label).astype(int)
    
    return X,Y


def genTrainDataBeta(g_nx,betaVec,sbeta,lambda_,slambda, initial_nodes, batch_size, timesteps):   
    #choose your beta and put it as label
    n_nodes = len(g_nx)
    X = np.zeros([batch_size, n_nodes, timesteps], np.int32)
    lambdaVec = np.abs(np.random.normal(lambda_, lambda_/10, batch_size))

    Y = zip(betaVec,lambdaVec)

    for i,(b,l)in enumerate(Y):
        X[i] = generateInfectionMatrix(g_nx,timesteps = timesteps,  beta=b, lambda_=l, initial_nodes = initial_nodes)

    #Y = np.array(list(Y))   
    #Y = (np.ones((batch_size,1))*label).astype(int)
    
    return X,betaVec

def genTrainDataUniform(g_nx,minBeta,maxBeta,minLambda,maxLambda, initial_nodes, batch_size, timesteps,label):   
    
    n_nodes = len(g_nx)
    X = np.zeros([batch_size, n_nodes, timesteps], np.int32)

    betaVec   = np.random.uniform(minBeta, maxBeta, batch_size)
    lambdaVec = np.random.uniform(minLambda, maxLambda, batch_size)

    Y = zip(betaVec,lambdaVec)

    for i,(b,l)in enumerate(Y):
        X[i] = generateInfectionMatrix(g_nx,timesteps = timesteps,  beta=b, lambda_=l, initial_nodes = initial_nodes)

    #Y = np.array(list(Y))   
    Y = (np.ones((batch_size,1))*label).astype(int)
    
    return X,Y

def epidemics_generator(g_nx,batch_size,timesteps,initial_nodes):

	print('\nGenerating epidemics - Manu\n\n')
	beta, lambda_ = 0.005, 0.005
	sbeta = beta/100
	slambda = lambda_/100
	x1,y1 = genTrainData(g_nx,beta,sbeta,lambda_,slambda, initial_nodes, int(batch_size/2), timesteps,label=1)

	#beta = 0.005
	minBeta,maxBeta = 0.000,0.020
	minLambda,maxLambda = 0.000,0.020
	#lambda_ = 0.005
	#slambda = lambda_/100
	#x2,y2 = genTrainDataUniform(g_nx,minBeta,maxBeta,minLambda,maxLambda, initial_nodes, int(batch_size/2), timesteps,label=0)
	x2,y2 = genTrainData(g_nx,0.002,sbeta,lambda_,slambda, initial_nodes, int(batch_size/2), timesteps,label=0)

	#concatenate
	X = np.concatenate((x1,x2),0)
	Y = np.concatenate((y1,y2),0)

	#shuffle

	assert len(X) == batch_size

	arr = np.arange(batch_size)
	np.random.shuffle(arr)
	X = X[arr]
	Y = Y[arr]

	X = np.expand_dims(X,3)
	Y = np.squeeze(Y)

	print('\n \n Epidemics Data Generated. Shape: {}'.format(X.shape))
	print(Y.shape)

	return X.astype('uint8'),Y.astype('uint8')

def epidemics_generator_mse(g_nx,batch_size,timesteps,initial_nodes):

	print('\nGenerating MSE epidemics\n\n')
	lambda_ = 0.005
	sbeta = 0.0000
	slambda = 0#lambda_/100
	minBeta,maxBeta = 0.000,0.05

	betaVec   = np.random.uniform(minBeta, maxBeta, batch_size)

	X,Y = genTrainDataBeta(g_nx,betaVec,sbeta,lambda_,slambda, initial_nodes, batch_size, timesteps)

	assert len(X) == batch_size

	arr = np.arange(batch_size)
	np.random.shuffle(arr)
	X = X[arr]
	Y = Y[arr]

	X = np.expand_dims(X,3)
	Y = np.squeeze(Y)

	print('\n \n Epidemics Data Generated. Shape: {}'.format(X.shape))
	print(Y.shape)

	return X.astype('uint8'),Y

def load_from_npy(path,num):

	X,Y = np.load(path+'Data.npy').astype(int),np.load(path+'Labels.npy').astype(int)

	arr = np.arange(len(X))
	np.random.shuffle(arr)
	X = X[arr]
	Y = Y[arr]

	return X[:num],Y[:num]

def save_to_npy(path,X,Y):
	np.save(path+'Data.npy',X)
	np.save(path+'Labels.npy',Y)
