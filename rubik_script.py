import random
import numpy as np
import pickle
import itertools


''' INITIALIZATION '''

n_cube = 2	# n x n x n cube (2 or 3)
n_hlayers = 2	# number of hidden layers (1,2 or 3)
H1 = 250	# size of first hidden layer
H2 = 250
H3 = 0
epnum = 0
batch_size = 100   # every how many episodes to do a param update?

shuffle_n = 100	# how many random actions in shuffling cube
actcount_max = 500	  # max amount of actions allowed
test_number = 'test18' # variable linked to file saving
zeros_fill = 5 # variable linked to file saving, the longer you run the larger needed

assert n_hlayers == np.count_nonzero([H1,H2,H3]), "n_hlayers and the H:s must match"

''' FILING AND PRACTICAL VARIABLES '''

print_every_batch = True
dump_values_and_net = False
end_at_episode = 1000000	 # only if dump_values_and_net == TRUE

resume = False # resume previous run?
if resume :
	epnum = 800
	resume_file = 'net' + str(n_cube) + '_h' + str(n_hlayers) + '_' + test_number + '_' + str(H1) + '.' + str(H2) + '.' + str(H3) + '_' \
	+ str(int(end_at_episode/1000)) + 'k-PU' + str(int(epnum/batch_size)).zfill(zeros_fill) + '.p'	# 'rubik3net-h2test01-E0001k.p'
	

# file saving
net_dump_file = 'net' + str(n_cube) + '_h' + str(n_hlayers) + '_' + test_number + '_' + str(H1) + '.' + str(H2) + '.' + str(H3) + '_' \
	+ str(int(end_at_episode/1000)) + 'k-PU'	# parameter update number "str(int(epnum / batch_size)).zfill(4)+'k.p'"
values_dump_file = 'values' + str(n_cube) + '_h' + str(n_hlayers) + '_' + test_number + '_' + str(H1) + '.' + str(H2) + '.' + str(H3) + '_' \
	+ str(int(end_at_episode/1000)).zfill(zeros_fill) + 'k.p'

	
''' CODE FOR CUBE '''

# cube is encoded as Piece -> Location, (every cubie (piece) is in a cubicle (location))
# W on top (Y bottom), R on left (O right), B in front (G back)

dctX = {-1:'r',0:' ',1:'o'}
dctY = {-1:'b',0:' ',1:'g'}
dctZ = {-1:'y',0:' ',1:'w'}
dctCol = {'r':(1,0,0,0,0,0), 'o':(0,1,0,0,0,0),
			'b':(0,0,1,0,0,0), 'g':(0,0,0,1,0,0),
			'y':(0,0,0,0,1,0), 'w':(0,0,0,0,0,1)}

poslist = [ np.array([x*(4-n_cube)-1, y*(4-n_cube)-1, z*(4-n_cube)-1]) for x in range(n_cube) for y in range(n_cube) for z in range(n_cube) ]

colors = [(dctX[p[0]]+dctY[p[1]]+dctZ[p[2]])
			for p in poslist]

solvedCube = [ (pos, pos*(1,2,3)) for pos in poslist]

def cubesides(cu) : # list the colors of the cube
	cupieces = [p.tolist() for p, o in cu]
	permu = [ cupieces.index(ps.tolist()) for ps in poslist ]
	jsL = [j for j in range(n_cube**3) if poslist[j][0]==-1]
	jsR = [j for j in range(n_cube**3) if poslist[j][0]== 1]
	jsF = [j for j in range(n_cube**3) if poslist[j][1]==-1]
	jsB = [j for j in range(n_cube**3) if poslist[j][1]== 1]
	jsD = [j for j in range(n_cube**3) if poslist[j][2]==-1]
	jsU = [j for j in range(n_cube**3) if poslist[j][2]== 1]
	return [
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[0]))-1] for j in jsL] ,
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[0]))-1] for j in jsR] ,
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[1]))-1] for j in jsF] ,
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[1]))-1] for j in jsB] ,
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[2]))-1] for j in jsD] ,
		[ (colors[permu[j]])[int(abs((cu[permu[j]][1])[2]))-1] for j in jsU] ,
		]

def printcube(cu) :
	sds = cubesides(cu)
	for s in range(len(sds)) :
		print('		  '+' '.join(sds[s])+'		 ('+('robgyw')[s]+')')


def score_for_cube(cu) :
	return [ p1==p2 for p1, p2 in
		zip([c for side in cubesides(cu) for c in side],
			[c for side in cubesides(solvedCube) for c in side])
				].count(True)


def rotZ(pos,sign) :
	return np.array([-sign*pos[1], sign*pos[0], pos[2]])

def rotX(pos,sign) :
	return np.array([pos[0], -sign*pos[2], sign*pos[1]])

def rotY(pos,sign) :
	return np.array([sign*pos[2], pos[1], -sign*pos[0]])


# Rotations: c : clockwise	  && a : anti-clockwise / against clockwise , looking at the face of the side.

def posoriApplyUc(pos, ori) :
	if(pos[2] != +1) : return pos, ori
	else : return rotZ(pos,-1), rotZ(ori,-1)

def posoriApplyUa(pos, ori) :
	if(pos[2] != +1) : return pos, ori
	else : return rotZ(pos,+1), rotZ(ori,+1)

def posoriApplyDa(pos, ori) :
	if(pos[2] != -1) : return pos, ori
	else : return rotZ(pos,-1), rotZ(ori,-1)

def posoriApplyDc(pos, ori) :
	if(pos[2] != -1) : return pos, ori
	else : return rotZ(pos,+1), rotZ(ori,+1)


def posoriApplyRc(pos, ori) :
	if(pos[0] != +1) : return pos, ori
	else : return rotX(pos,-1), rotX(ori,-1)

def posoriApplyRa(pos, ori) :
	if(pos[0] != +1) : return pos, ori
	else : return rotX(pos,+1), rotX(ori,+1)

def posoriApplyLa(pos, ori) :
	if(pos[0] != -1) : return pos, ori
	else : return rotX(pos,-1), rotX(ori,-1)

def posoriApplyLc(pos, ori) :
	if(pos[0] != -1) : return pos, ori
	else : return rotX(pos,+1), rotX(ori,+1)


def posoriApplyBc(pos, ori) :
	if(pos[1] != +1) : return pos, ori
	else : return rotY(pos,-1), rotY(ori,-1)

def posoriApplyBa(pos, ori) :
	if(pos[1] != +1) : return pos, ori
	else : return rotY(pos,+1), rotY(ori,+1)

def posoriApplyFa(pos, ori) :
	if(pos[1] != -1) : return pos, ori
	else : return rotY(pos,-1), rotY(ori,-1)

def posoriApplyFc(pos, ori) :
	if(pos[1] != -1) : return pos, ori
	else : return rotY(pos,+1), rotY(ori,+1)


# Rotations: c : clockwise	  && a : anti-clockwise / against clockwise , looking at the face of the side.

def posoriApplyNone(ps,oi) :
	return ps, oi

def applyNone(cu) :
	return cu

def applyLc (cu) :
	return [posoriApplyLc(c[0],c[1]) for c in cu]

def applyLa (cu) :
	return [posoriApplyLa(c[0],c[1]) for c in cu]

def applyRc (cu) :
	return [posoriApplyRc(c[0],c[1]) for c in cu]

def applyRa (cu) :
	return [posoriApplyRa(c[0],c[1]) for c in cu]

def applyFc (cu) :
	return [posoriApplyFc(c[0],c[1]) for c in cu]

def applyFa (cu) :
	return [posoriApplyFa(c[0],c[1]) for c in cu]

def applyBc (cu) :
	return [posoriApplyBc(c[0],c[1]) for c in cu]

def applyBa (cu) :
	return [posoriApplyBa(c[0],c[1]) for c in cu]

def applyDc (cu) :
	return [posoriApplyDc(c[0],c[1]) for c in cu]

def applyDa (cu) :
	return [posoriApplyDa(c[0],c[1]) for c in cu]

def applyUc (cu) :
	return [posoriApplyUc(c[0],c[1]) for c in cu]

def applyUa (cu) :
	return [posoriApplyUa(c[0],c[1]) for c in cu]


def shuffle(cu, moves=1) :
	retcu = cu.copy()
	for _ in range(moves) :
		mv = random.choice([applyNone,
			applyLc, applyLa, applyRc, applyRa,
			applyFc, applyFa, applyBc, applyBa,
			applyDc, applyDa, applyUc, applyUa])
		retcu = mv(retcu)
	return retcu



def prepro(cu) : # preprosessing of cube
	codes = itertools.chain.from_iterable([dctCol[si]
		for si in itertools.chain.from_iterable(cubesides(cu))])
	return ( np.array([ c for c in codes])-1/6*np.ones(6*6*n_cube**2) )


	
	

'''REINFORCEMENT LEARNING'''

# hyperparameters

learning_rate = 8e-4
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
gamma = 0.98 # discount factor for reward


# model initialization

Din = 6*6*n_cube**2	 # input dimensionality: 6 colors 6 sides of 3x3 grids
Dout = 13 # decision dimensionality: 12 possible moves and 1 option to stop

# weights
if resume:
	rnet = pickle.load(open(resume_file, 'rb'))
else:
	rnet = {}
	rnet['W1'] = np.random.randn(H1,Din) / np.sqrt(Din)
	if H2 != 0 :
		rnet['W2'] = np.random.randn(H2,H1) / np.sqrt(H1)
		if H3 != 0 :
			rnet['W3'] = np.random.randn(H3,H2) / np.sqrt(H2)
			rnet['W4'] = np.random.randn(Dout,H3) / np.sqrt(H3)
		else :
			rnet['W3'] = np.random.randn(Dout,H2) / np.sqrt(H2)
			rnet['W4'] = []
	else :
		rnet['W2'] = np.random.randn(Dout,H1) / np.sqrt(H1)
		rnet['W3'] = []
		rnet['W4'] = []

grad_buffer = { k : np.zeros_like(v) for k,v in rnet.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in rnet.items() } # rmsprop memory
	
	
def softmax(xvec):
	exps = [np.exp(xval) for xval in xvec]
	return exps / sum(exps)

def feedforward(x):
	h1 = np.dot(rnet['W1'], x)
	h1[h1<0] = 0 # ReLU nonlinearity
	h2, h3 = [], []
	if H2 != 0 :
		h2 = np.dot(rnet['W2'], h1)
		h2[h2<0] = 0 # ReLU nonlinearity
		if H3 != 0 :
			h3 = np.dot(rnet['W3'], h2)
			h3[h3<0] = 0 # ReLU nonlinearity
			logp = np.dot(rnet['W4'], h3)
		else :
			logp = np.dot(rnet['W3'], h2)
	else :
		logp = np.dot(rnet['W2'], h1)
	p = softmax(logp)
	return p, h1, h2, h3 # return probabilities of actions and hidden states

def backprop(eph1, eph2, eph3, epx, epdlogp): #backpropagation
	""" backward pass. (eph is array of intermediate hidden states) """
	if H3 != 0 :
		dW4 = np.dot(epdlogp.T,eph3)
		
		W4 = (rnet['W4']).copy()
		dh3 = np.dot(epdlogp, W4)
		dh3[eph3 <= 0] = 0	  # d_preaktivaatio
		dW3 = np.dot(dh3.T, eph2)
		
		W3 = (rnet['W3']).copy()
		dh2 = np.dot(dh3, W3)
		dh2[eph2 <= 0] = 0	  # d_preaktivaatio
		dW2 = np.dot(dh2.T, eph1)
		
		W2 = (rnet['W2']).copy()
		dh1 = np.dot(dh2, W2)
		dh1[eph1 <= 0] = 0	  # d_preaktivaatio
		dW1 = np.dot(dh1.T, epx)
	
	
	elif H2 != 0 :
		dW3 = np.dot(epdlogp.T,eph2)
		
		W3 = (rnet['W3']).copy()
		dh2 = np.dot(epdlogp, W3)
		dh2[eph2 <= 0] = 0	  # d_preaktivaatio
		dW2 = np.dot(dh2.T, eph1)
		
		W2 = (rnet['W2']).copy()
		dh1 = np.dot(dh2, W2)
		dh1[eph1 <= 0] = 0	  # d_preaktivaatio
		dW1 = np.dot(dh1.T, epx)
		
		dW4 = []
		
	else :
		dW2 = np.dot(epdlogp.T,eph1)
		
		W2 = (rnet['W2']).copy()
		dh1 = np.dot(epdlogp, W2)
		dh1[eph1 <= 0] = 0	  # d_preaktivaatio
		dW1 = np.dot(dh1.T, epx)
		
		dW3 = []
		dW4 = []
		
	return {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4':dW4}


actlist = [applyNone,
			applyLc, applyLa, applyRc, applyRa,
			applyFc, applyFa, applyBc, applyBa,
			applyDc, applyDa, applyUc, applyUa]
actinvlist = [applyNone,
			applyLa, applyLc, applyRa, applyRc,
			applyFa, applyFc, applyBa, applyBc,
			applyDa, applyDc, applyUa, applyUc]
onestepcubes=[act(solvedCube) for act in actinvlist[1:]]

startcb = shuffle(solvedCube,moves=shuffle_n)
cb = startcb.copy()
xs, hs1, hs2, hs3, dlogps, cbs = [], [], [], [], [], []
actcount = 0

# parameters for the score (cost) function
score_baseline = 1.0
score_baseline_horizon = 1
max_horizon = 1000

score_history = []
duration_history = []
duration_history_mean = []
score_actual = []
score_actual_mean = []
percentage_solved = []
onestepsolhist= []
stopatsolvedhist= []


if epnum == 0 and dump_values_and_net: 
	pickle.dump(rnet, open(net_dump_file+str(int(epnum / batch_size)).zfill(zeros_fill)+'.p', 'wb'))
epnum += 1 # to prevent zero update


''' RUN AND UPDATE PARAMETERS '''

while True:
	# forward the policy network and sample an action from the returned probability
	x = np.array(prepro(cb))
	aprob, h1, h2, h3 = feedforward(x)
	action = np.random.choice(range(13), p=aprob) # stochastic policy

	# record various intermediates (needed later for backprop)
	xs.append(x) # observation
	hs1.append(h1) # hidden state
	hs2.append(h2)
	hs3.append(h3)
	y = np.zeros(13)
	y[action]=1.0
	dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

	# step the environment and get new measurements
	cb = (actlist[action])(cb)
	cbs.append(cb)
	actcount += 1

	if (action == 0 or actcount > actcount_max) : # an episode finished
		score = score_for_cube(cb)
		score_actual.append(score)
		if (score==n_cube**2*6) : # give a significant bonus for fully correct solution
			score += 100
		
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		epx = np.vstack(xs)
		eph1 = np.vstack(hs1)
		eph2 = np.vstack(hs2)
		eph3 = np.vstack(hs3)
		epdlogp = np.vstack(dlogps)

		# compute the standardized (discounted?) reward
		score_history.append(score)
		duration_history.append(actcount)
		score -= score_baseline
		if score_baseline_horizon > 1 :
			score /= (.1+np.std(score_history[-score_baseline_horizon:]))
		score_baseline_horizon = min(score_baseline_horizon+1,max_horizon)
		score_baseline = np.mean(score_history[-score_baseline_horizon:])

		eprew = np.vstack(np.array([score for _ in range(actcount)])
				* np.array([gamma**(actcount-t-1) for t in range(actcount)]))

		# reset array memory
		xs, hs1, hs2, hs3, dlogps, cbs = [], [], [], [], [], []
		actcount=0

		epdlogp *= eprew # modulate the gradient with advantage (PG magic happens right here.)
		grad = backprop(eph1, eph2, eph3, epx, epdlogp)
		for k in rnet: grad_buffer[k] += grad[k] # accumulate grad over batch

		# perform rmsprop PARAMETER UPDATE every batch_size episodes
		if (epnum % batch_size == 0) :
			for k,v in rnet.items():
				g = grad_buffer[k] # gradient
				rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
				rnet[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
				grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
				
			# record different quantities
			percentage_solved.append(len([ x for x in score_actual[-batch_size:] if x == n_cube**2*6 ]) / batch_size ) # percentage of last 100 solved
			score_actual_mean.append(np.mean(score_actual[-batch_size:]))
			duration_history_mean.append(np.mean(duration_history[-batch_size:]))
			onestepsolhist.append([feedforward(np.array(prepro(onestepcubes[i])))[0][i+1]
				for i in range(len(onestepcubes))])
			stopatsolvedhist.append(feedforward(np.array(prepro(solvedCube)))[0][0])

			if print_every_batch :
				print("ep {} , current scores ~ {} , attempt moves ~ {}".format(
					epnum, np.mean(score_history[-score_baseline_horizon:]), #np.std(score_history[-score_baseline_horizon:]
					np.mean(duration_history[-score_baseline_horizon:])))
				print("	 stop at solved = 1 - {:.7f}".format(1 - stopatsolvedhist[-1]))
				print("	 success at one step = {:.3f}".format(
						np.mean(onestepsolhist[-1])))
				print("			  median = {:.3f}	   max = {:.3f}		 min = {:.8f}".format(
						np.median(onestepsolhist[-1]),
						max(onestepsolhist[-1]),min(onestepsolhist[-1])))
				print("Example finished state (score {}):".format(score_for_cube(cb)))
				printcube(cb)
			
			if dump_values_and_net : 
				pickle.dump(rnet, open(net_dump_file+str(int(epnum / batch_size)).zfill(zeros_fill)+'.p', 'wb'))
				if epnum == end_at_episode :
					output = dict(zip(['onestep','solved','percentage_solved','score_mean','dur_mean'],
						[onestepsolhist,stopatsolvedhist,percentage_solved,score_actual_mean,duration_history_mean]))
					pickle.dump(output,open(values_dump_file,'wb'))
					print("terminating.")
					break
					
			if epnum % 1000 == 0 : # automatic printing for progress monitoring
				print("	 Ep: {:7.0f}, mean(for latest batch): {:4.2f}, max score(b): {}, N solved: {:.0f}"
					  .format(epnum, score_actual_mean[-1], np.max(score_actual[-batch_size:]), percentage_solved[-1] * batch_size))
			score_actual, duration_history = [], []

		# reshuffle cube
		epnum += 1
		startcb = shuffle(solvedCube,moves=shuffle_n)
		cb = startcb.copy()
