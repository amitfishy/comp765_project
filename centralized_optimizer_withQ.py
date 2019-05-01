import numpy as np
import time

from scipy.optimize import linprog
from cvxopt import matrix, solvers, spmatrix

from envs.multi_robot_task_basic import mrt_basic





# env = mrt_basic(random_seed = 12)
# # print (env.observation_space.n)
# # print (env.action_space.n)
# # print (env.P)

# num_states = env.observation_space.n
# num_actions = env.action_space.n
# gamma = 0.999


# def do_opposite(action):
# 	opp_action = [-1, -1]
# 	if action[0] == 0:
# 		opp_action[0] = 1
# 	else:
# 		opp_action[0] = 0

# 	if action[1] == 0:
# 		opp_action[1] = 1
# 	else:
# 		opp_action[1] = 0
# 	return opp_action

def get_optimal_action(Q_opt, obs, num_actions):
	opt_action = np.argmin(Q_opt[obs*num_actions:obs*num_actions + num_actions])

	# print(opt_action)

	bin_action = "{0:b}".format(opt_action)
	if len(bin_action) == 1:
		action1 = 0
		action2 = int(bin_action)
	else:
		action1 = int(bin_action[0])
		action2 = int(bin_action[1])

	# print ('BEFORE: ', action1, action2)

	# if action1==0:
	# 	action1=1
	# else:
	# 	action1=0

	# if action2==0:
	# 	action2=1
	# else:
	# 	action2=0

	# print ('AFTER: ', action1, action2)

	return [action1, action2]

def perform_centralized_optimization(env, num_states, num_actions, gamma):
	Q = np.ones(num_states * num_actions) / float(num_states)
	J = np.ones(num_states)


	objective = np.concatenate((Q, 	np.zeros_like(J)))

	constraints_lhs = np.zeros((2*num_states*num_actions,) + objective.shape)
	constraints_rhs = np.zeros(2*num_states*num_actions)

	counter = 0
	for ns in range(num_states):
		for na in range(num_actions):
			constraints_lhs[counter, ns*num_actions + na] = 1
			for tup in env.P[ns][na]:
				prob = tup[0]
				next_ns = tup[1]
				constraints_lhs[counter, num_states*num_actions + next_ns] = -gamma * prob

			counter = counter + 1

	for ns in range(num_states):
		for na in range(num_actions):
			constraints_lhs[counter, ns*num_actions + na] = -1
			constraints_lhs[counter, num_states*num_actions + ns] = 1

			counter = counter + 1

	counter = 0
	for ns in range(num_states):
		for na in range(num_actions):
			constraints_rhs[counter] = -env.P[ns][na][0][2]

			counter = counter + 1

	#res = linprog(-objective, A_ub=constraints_lhs, b_ub=constraints_rhs, options={"disp": True, "sym_pos": False}, method='interior-point')

	c = matrix(-objective)
	A = matrix(constraints_lhs.transpose().tolist())
	b = matrix(constraints_rhs)

	# print (constraints_lhs[-4:,:])
	# print (A)

	# exit()

	# print (constraints_lhs.shape)

	# print (solvers.options)
	solvers.options['maxiters'] = 1000
	# solvers.options['feastol']=1e-7
	sol = solvers.lp(c,A,b, solver='glpk')



	# print (sol['x'])
	# print (sol['status'])
	x = np.array(sol['x']).reshape(-1)
	# print(x)

	return x[:-num_states], x[-num_states:]



if __name__ == "__main__":
	env = mrt_basic(random_seed = 12)
	# print (env.observation_space.n)
	# print (env.action_space.n)
	# print (env.P)

	num_states = env.observation_space.n
	num_actions = env.action_space.n
	gamma = 0.999

	Q_func, J_func = perform_centralized_optimization(env, num_states, num_actions, gamma)

	obs,r,_,_ = env.step([0,0])
	env.render()
	time.sleep(1)
	cum_rew = r

	for i in range(200):
		print (i)
		
		print('Obs: ', obs)
		action = get_optimal_action(Q_func, obs, num_actions)


		obs,r,_,_ = env.step(action)
		print('Action: ', action)

		cum_rew = cum_rew + r
		print ('Current cumulative reward: ', cum_rew)
		
		env.render()
		time.sleep(1)