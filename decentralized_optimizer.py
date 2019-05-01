import numpy as np
import time
from cvxopt import matrix, solvers, spmatrix

from envs.multi_robot_task_basic import mrt_basic

import matplotlib.pyplot as plt

NUM_MAIN_ITERS = 5
NUM_STEPS_RUN = 2000

def get_optimal_action(Q_functions, obs, env, num_states, num_actions):
	task_num, agent1_status, agent2_status = env.decode(obs)
	ag1_state = env.encode_decentralized(task_num, agent1_status)
	ag2_state = env.encode_decentralized(task_num, agent2_status)

	Q_agent1 = Q_functions[0:round(num_states * num_actions / 4)]
	Q_agent2 = Q_functions[round(num_states * num_actions / 4): round(num_states * num_actions / 2)]

	# print(Q_agent1[ag1_state*round(num_actions/2) : ag1_state*round(num_actions/2) + round(num_actions/2)])
	# print(Q_agent2[ag2_state*round(num_actions/2) : ag2_state*round(num_actions/2) + round(num_actions/2)])
	action1 = np.argmin(Q_agent1[ag1_state*round(num_actions/2) : ag1_state*round(num_actions/2) + round(num_actions/2)])
	action2 = np.argmin(Q_agent2[ag2_state*round(num_actions/2) : ag2_state*round(num_actions/2) + round(num_actions/2)])
	return [action1, action2]

def get_separate_actions(action):
	bin_action = "{0:b}".format(action)
	if len(bin_action) == 1:
		action1 = 0
		action2 = int(bin_action)
	else:
		action1 = int(bin_action[0])
		action2 = int(bin_action[1])

	return action1, action2

def perform_decentralized_optimization(state_dist, env, num_states, num_actions, gamma):

	objective = np.zeros(round(num_states * num_actions / 2) + num_states)
	for ns in range(num_states):
		for na in range(num_actions):
			task_num, agent1_status, agent2_status = env.decode(ns)
			ag1_state = env.encode_decentralized(task_num, agent1_status)
			ag2_state = env.encode_decentralized(task_num, agent2_status)
			action1, action2 = get_separate_actions(na)

			objective[ag1_state*round(num_actions/2) + action1] = objective[ag1_state*round(num_actions/2) + action1] + float(state_dist[ns])
			objective[round(num_states * num_actions / 4) + ag2_state*round(num_actions/2) + action2] = objective[round(num_states * num_actions / 4) + ag2_state*round(num_actions/2) + action2] + float(state_dist[ns])

	# objective = objective / float(num_states)

	lhs = np.zeros((2 * num_states * num_actions,) + objective.shape)
	rhs = np.zeros(2 * num_states * num_actions)


	counter = 0
	for ns in range(num_states):
		for na in range(num_actions):
			task_num, agent1_status, agent2_status = env.decode(ns)
			ag1_state = env.encode_decentralized(task_num, agent1_status)
			ag2_state = env.encode_decentralized(task_num, agent2_status)
			action1, action2 = get_separate_actions(na)

			lhs[counter, ag1_state*round(num_actions/2) + action1] = 1
			lhs[counter, round(num_states * num_actions / 4) + ag2_state*round(num_actions/2) + action2] = 1

			for tuple_list in env.P[ns][na]:
				prob = tuple_list[0]
				next_state = tuple_list[1]
				task_num, agent1_status, agent2_status = env.decode(next_state)
				ag1_state = env.encode_decentralized(task_num, agent1_status)
				ag2_state = env.encode_decentralized(task_num, agent2_status)

				lhs[counter, round(num_states * num_actions / 2) + ag1_state] = lhs[counter, round(num_states * num_actions / 2) + ag1_state] - (gamma * prob)
				lhs[counter, round(num_states * num_actions / 2) + round(num_states / 2) + ag2_state] = lhs[counter, round(num_states * num_actions / 2) + round(num_states / 2) + ag2_state] - (gamma * prob)

			rhs[counter] = -env.P[ns][na][0][2]

			counter = counter + 1

	for ns in range(num_states):
		for na in range(num_actions):
			task_num, agent1_status, agent2_status = env.decode(ns)
			ag1_state = env.encode_decentralized(task_num, agent1_status)
			ag2_state = env.encode_decentralized(task_num, agent2_status)
			action1, action2 = get_separate_actions(na)

			lhs[counter, ag1_state*round(num_actions/2) + action1] = -1
			lhs[counter, round(num_states * num_actions / 4) + ag2_state*round(num_actions/2) + action2] = -1

			lhs[counter, round(num_states * num_actions / 2) + ag1_state] = 1
			lhs[counter, round(num_states * num_actions / 2) + round(num_states / 2) + ag2_state] = 1

			counter = counter + 1

	c = matrix(-objective)
	G = matrix(lhs.transpose().tolist())
	b = matrix(rhs)

	solvers.options['maxiters'] = 1000
	sol = solvers.lp(c,G,b, solver='glpk')

	# print (sol['status'])
	x = np.array(sol['x']).reshape(-1)
	# print(x)


	env.reset()

	obs,r,_,_ = env.step([0,0])
	cum_rew = r

	result_state_dist = np.zeros(num_states)

	for i in range(NUM_STEPS_RUN):
		# print (i)

		action = get_optimal_action(x[:-num_states], obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)
		result_state_dist[obs] = result_state_dist[obs] + 1
		# print('Action: ', action)

		cum_rew = cum_rew + r

		# print ('Current cumulative reward: ', cum_rew)
		
		# env.render()
		# time.sleep(1)

	result_state_dist = result_state_dist / float(NUM_STEPS_RUN)
	# print (result_state_dist)
	# plt.bar([i for i in range(num_states)], result_state_dist)
	# plt.show()

	return result_state_dist, x[:-num_states], x[-num_states:]


# def get_w_dist(Q_func):
# 	init_distribution = np.zeros(num_states)
# 	uniform_dist_value = 1. / float(env.num_tasks + 1)
# 	for i in range(env.num_tasks + 1):
# 		init_distribution[env.encode(i, 0, 0)] = uniform_dist_value

# 	P_mu = np.zeros((num_states, num_states))

# 	for from_state in range(num_states):
# 		for to_state in range(num_states):
# 			action = get_optimal_action(Q_func, from_state)
# 			action = action[0]*2 +action[1]
# 			for tuple_list in env.P[from_state][action]:
# 				prob = tuple_list[0]
# 				next_state = tuple_list[1]
# 				if to_state == next_state:
# 					P_mu[from_state, to_state] = P_mu[from_state, to_state] + prob

# 	w_dist = (1 - gamma) * np.matmul(np.linalg.inv(np.eye(num_states) - gamma*P_mu), init_distribution)
# 	return w_dist


if __name__ == "__main__":
	env = mrt_basic(random_seed = 12)

	num_states = env.observation_space.n
	num_actions = env.action_space.n
	gamma = 0.999

	state_dist = np.ones(num_states) / float(num_states)
	for i in range(NUM_MAIN_ITERS):
		# print ('Current Iteration: ' + str(i+1))
		state_dist, Q_func, J_func = perform_decentralized_optimization(state_dist, env, num_states, num_actions, gamma)
		# msdiff = (np.sum((next_state_dist - state_dist)**2) / num_states)**0.5
		# print ('Difference between state distributions: ', msdiff)
		# state_dist = get_w_dist(Q_func)

	# plt.bar([i for i in range(num_states)], state_dist)
	# plt.show()


	# #per task visualizer
	# print (env.task_locations)
	# env.render()
	# time.sleep(1)

	# for j in range(10):
	# 	print ('-----------------Repeating Now----------------------')
	# 	for i in range(env.num_tasks + 1):
	# 		env.set_state(i)
	# 		obs = env.state

	# 		action = get_optimal_action(Q_func, obs, env, num_states, num_actions)
	# 		next_obs,r,_,_ = env.step(action)
	# 		print('Action: ', action)
	# 		if i!=0:
	# 			print(env.task_locations[i-1])
	# 		env.render()
	# 		time.sleep(1)
	# 	time.sleep(3)



	# #normal visualizer
	# obs,r,_,_ = env.step([0,0])
	# env.render()
	# time.sleep(1)
	# cum_rew = r

	# for i in range(200):
	# 	print (i)
		
	# 	print('Obs: ', obs)
	# 	action = get_optimal_action(Q_func, obs, env, num_states, num_actions)


	# 	obs,r,_,_ = env.step(action)
	# 	print('Action: ', action)

	# 	cum_rew = cum_rew + r
	# 	print ('Current cumulative reward: ', cum_rew)
		
	# 	env.render()
	# 	time.sleep(1)



	#normal visualizer
	env = mrt_basic(random_seed = 12)

	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i in range(NUM_STEPS_RUN):
		action = get_optimal_action(Q_func, obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)

		cum_rew = cum_rew + r

	print(cum_rew)