import numpy as np
import time

from scipy.optimize import linprog
from cvxopt import matrix, solvers, spmatrix

from envs.multi_robot_task_basic import mrt_basic


env = mrt_basic(random_seed = 103)

num_states = env.observation_space.n
num_actions = env.action_space.n
gamma = 0.999


def get_optimal_action(Q_opt, obs):
	opt_action = np.argmin(Q_opt[obs, :])

	bin_action = "{0:b}".format(opt_action)
	if len(bin_action) == 1:
		action1 = 0
		action2 = int(bin_action)
	else:
		action1 = int(bin_action[0])
		action2 = int(bin_action[1])
	return [action1, action2]

def get_optimal_Q(optimal_J):
	optimal_Q = np.zeros((num_states, num_actions))

	counter = 0
	for ns in range(num_states):
		for na in range(num_actions):
			optimal_Q[ns, na] = -env.P[ns][na][0][2]
			for tup in env.P[ns][na]:
				prob = tup[0]
				next_ns = tup[1]
				optimal_Q[ns, na] = optimal_Q[ns, na] + gamma*prob*optimal_J[next_ns]

	return optimal_Q


def perform_centralized_optimization():
	objective = np.ones(num_states) / float(num_states)
	constraints_lhs = np.zeros((num_states*num_actions,) + objective.shape)
	constraints_rhs = np.zeros(num_states*num_actions)

	counter = 0
	for ns in range(num_states):
		for na in range(num_actions):
			constraints_lhs[counter, ns] = 1
			for tup in env.P[ns][na]:
				prob = tup[0]
				next_ns = tup[1]
				constraints_lhs[counter, next_ns] = constraints_lhs[counter, next_ns] - gamma * prob
			constraints_rhs[counter] = -env.P[ns][na][0][2]
			counter = counter + 1

	c = matrix(-objective)
	A = matrix(constraints_lhs.transpose().tolist())
	b = matrix(constraints_rhs)

	solvers.options['maxiters'] = 1000
	sol = solvers.lp(c,A,b, solver='glpk')

	print (sol['status'])
	x = np.array(sol['x']).reshape(-1)
	print(x)

	opt_Q = get_optimal_Q(x)
	print(opt_Q)

	return opt_Q
	



if __name__ == "__main__":
	opt_Q = perform_centralized_optimization()

	obs,r,_,_ = env.step([0,0])
	env.render()
	time.sleep(1)
	cum_rew = r

	for i in range(200):
		print (i)
		
		print('Obs: ', obs)
		action = get_optimal_action(opt_Q, obs)


		obs,r,_,_ = env.step(action)
		print('Action: ', action)

		cum_rew = cum_rew + r
		print ('Current cumulative reward: ', cum_rew)
		
		env.render()
		time.sleep(1)