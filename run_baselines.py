import matplotlib.pyplot as plt
import numpy as np

import decentralized_optimizer as d_o
import centralized_optimizer_withQ as c_o


from envs.multi_robot_task_basic import mrt_basic

NUM_ENV_RUNS = 100
NUM_ENV_ITERS = 1000

cum_rews_c = []
cum_rews_d = []
cum_rews_always_resp = []
cum_rews_closest_resp = []

for i_ner in range(NUM_ENV_RUNS):
	random_seed = np.random.randint(0,1000)
	env = mrt_basic(random_seed = random_seed)

	num_states = env.observation_space.n
	num_actions = env.action_space.n
	gamma = 0.5
	state_dist = np.ones(num_states) / float(num_states)

	#decentralized
	state_dist, Q_func_d, J_func_d = d_o.perform_decentralized_optimization(state_dist, env, num_states, num_actions, gamma)

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


	#ALWAYS RESPOND
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i in range(NUM_ENV_ITERS):
		obs,r,_,_ = env.step([1,1])
		cum_rew = cum_rew + r

	cum_rews_always_resp.append(-cum_rew)


	#CLOSEST RESPOND
	env = mrt_basic(random_seed = random_seed)
	obs,r,_,_ = env.step([0,0])
	cum_rew = 0

	for i in range(NUM_ENV_ITERS):
		task_num, _, _ = env.decode(obs)
		
		if task_num != 0:
			dist_agent1 = (env.task_locations[task_num-1][0]**2 + env.task_locations[task_num-1][1]**2)**0.5
			dist_agent2 = ((env.task_locations[task_num-1][0] - 1)**2 + (env.task_locations[task_num-1][1]-1)**2)**0.5
			if dist_agent1 < dist_agent2:
				action = [1,0]
			else:
				action = [0,1]
		else:
			action = [0,0]

		obs,r,_,_ = env.step(action)
		cum_rew = cum_rew + r

	cum_rews_closest_resp.append(-cum_rew)

print('DEC')
# print(cum_rews_d)
print(np.mean(cum_rews_d))
print(np.std(cum_rews_d))

print('CENT')
# print(cum_rews_c)
print(np.mean(cum_rews_c))
print(np.std(cum_rews_c))

print('ALWAYS')
# print(cum_rews_c)
print(np.mean(cum_rews_always_resp))
print(np.std(cum_rews_always_resp))

print('CLOSEST')
# print(cum_rews_c)
print(np.mean(cum_rews_closest_resp))
print(np.std(cum_rews_closest_resp))

# plt.title('ABC')
# plt.xlabel('States')
# plt.ylabel('Cost')
# plt.plot([i for i in range(num_states)], J_func_c, label='Centralized')
# plt.plot([i for i in range(num_states)], J_func_d, label='Decentralized')

# plt.legend(loc='upper left')
# plt.show()

# Build the plot
CTEs = [np.mean(cum_rews_d), np.mean(cum_rews_c), np.mean(cum_rews_always_resp), np.mean(cum_rews_closest_resp)]
error = [np.std(cum_rews_d), np.std(cum_rews_c), np.std(cum_rews_always_resp), np.std(cum_rews_closest_resp)]
fig, ax = plt.subplots()
ax.bar(np.arange(len(CTEs)), CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Cumulative Cost')
ax.set_xticks(np.arange(len(CTEs)))
ax.set_xticklabels(['DEC', 'CENT', 'ALWAYS', 'CLOSEST'])
ax.set_title('Performance of Different Policies')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()