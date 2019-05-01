import matplotlib.pyplot as plt
import numpy as np

import decentralized_optimizer as d_o
import centralized_optimizer_withQ as c_o


from envs.multi_robot_task_basic import mrt_basic

NUM_ENV_RUNS = 20
NUM_ENV_ITERS = 1000

cum_rews_c = []
cum_rews_d = []
cum_rews_d_5 = []
cum_rews_d_10 = []
cum_rews_d_20 = []

for i_ner in range(NUM_ENV_RUNS):
	random_seed = np.random.randint(0,1000)
	env = mrt_basic(random_seed = random_seed)

	num_states = env.observation_space.n
	num_actions = env.action_space.n
	gamma = 0.999
	

	#decentralized
	state_dist = np.ones(num_states) / float(num_states)
	state_dist, Q_func_d, J_func_d = d_o.perform_decentralized_optimization(state_dist, env, num_states, num_actions, gamma)



	#reset env
	env = mrt_basic(random_seed = random_seed)
	#decentralized 5
	state_dist_5 = np.ones(num_states) / float(num_states)
	for i in range(5):
		state_dist_5, Q_func_d_5, J_func_d_5 = d_o.perform_decentralized_optimization(state_dist_5, env, num_states, num_actions, gamma)



	#reset env
	env = mrt_basic(random_seed = random_seed)
	#decentralized 5
	state_dist_10 = np.ones(num_states) / float(num_states)
	for i in range(10):
		state_dist_10, Q_func_d_10, J_func_d_10 = d_o.perform_decentralized_optimization(state_dist_10, env, num_states, num_actions, gamma)



	#reset env
	env = mrt_basic(random_seed = random_seed)
	#decentralized 5
	state_dist_20 = np.ones(num_states) / float(num_states)
	for i in range(20):
		state_dist_20, Q_func_d_20, J_func_d_20 = d_o.perform_decentralized_optimization(state_dist_20, env, num_states, num_actions, gamma)



	#reset env
	env = mrt_basic(random_seed = random_seed)
	#centralized
	Q_func_c, J_func_c = c_o.perform_centralized_optimization(env, num_states, num_actions, gamma)


	#DECENTRALIZED
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i_nei in range(NUM_ENV_ITERS):
		action = d_o.get_optimal_action(Q_func_d, obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_d.append(-cum_rew)


	#CENTRALIZED
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i in range(NUM_ENV_ITERS):
		action = c_o.get_optimal_action(Q_func_c, obs, num_actions)
		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_c.append(-cum_rew)


	#DECENTRALIZED 5
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i_nei in range(NUM_ENV_ITERS):
		action = d_o.get_optimal_action(Q_func_d_5, obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_d_5.append(-cum_rew)


	#DECENTRALIZED 10
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i_nei in range(NUM_ENV_ITERS):
		action = d_o.get_optimal_action(Q_func_d_10, obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_d_10.append(-cum_rew)


	#DECENTRALIZED 20
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i_nei in range(NUM_ENV_ITERS):
		action = d_o.get_optimal_action(Q_func_d_20, obs, env, num_states, num_actions)
		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_d_20.append(-cum_rew)

print('CENT')
print(np.mean(cum_rews_c))
print(np.std(cum_rews_c))

print('DEC')
print(np.mean(cum_rews_d))
print(np.std(cum_rews_d))

print('DEC-5')
print(np.mean(cum_rews_d_5))
print(np.std(cum_rews_d_5))

print('DEC-10')
print(np.mean(cum_rews_d_10))
print(np.std(cum_rews_d_10))

print('DEC-20')
print(np.mean(cum_rews_d_20))
print(np.std(cum_rews_d_20))

# plt.title('ABC')
# plt.xlabel('States')
# plt.ylabel('Cost')
# plt.plot([i for i in range(num_states)], J_func_c, label='Centralized')
# plt.plot([i for i in range(num_states)], J_func_d, label='Decentralized')

# plt.legend(loc='upper left')
# plt.show()

# Build the plot
CTEs = [np.mean(cum_rews_c), np.mean(cum_rews_d), np.mean(cum_rews_d_5), np.mean(cum_rews_d_10), np.mean(cum_rews_d_20)]
error = [np.std(cum_rews_c), np.std(cum_rews_d), np.std(cum_rews_d_5), np.std(cum_rews_d_10), np.std(cum_rews_d_20)]
fig, ax = plt.subplots()
ax.bar(np.arange(len(CTEs)), CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Cumulative Cost')
ax.set_xticks(np.arange(len(CTEs)))
ax.set_xticklabels(['CENT', 'DEC', 'DEC-5', 'DEC-10', 'DEC-20'])
ax.set_title('Performance using Iteratively Obtained State Distributions')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()