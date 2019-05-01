from gym import spaces
import numpy as np

import cv2

BUSY_PROB = 0.8
DIST_MULT = 3
class mrt_basic():
	def __init__(self, num_tasks = 10, random_seed = 27):
		#initial state
		self.state = self.encode(np.random.randint(0, num_tasks+1), 0, 0)
		self.lastaction = None
		#Needed for render()
		self.laststate = None
		self.num_tasks = num_tasks

		#set random seed for reproducibility
		np.random.seed(random_seed)
		#create random task locations within the unit square
		#columns represent (x,y) and rows represent locations
		self.task_locations = np.random.uniform(0, 1, (num_tasks, 2))
		#(num_tasks locations + 1 no location = 11) X
		#(Agent1 is Busy or Agent1 is Free = 2)
		#(Agent2 is Busy or Agent2 is Free = 2)
		num_states = (num_tasks+1) * 2 * 2
		self.observation_space = spaces.Discrete(num_states)
		num_actions = 4
		#2 Agents: Each agent can Respond or NotRespond
		#multi agent actions are in order: {(0,0), (0,1), (1,0), (1,1)}
		self.action_space = spaces.Discrete(num_actions)

		self.P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}
		#task_num is the location ID, 0 means no location request
		#agent_status is 0 for Free and 1 for Busy
		for task_num in range(self.num_tasks+1):
			for agent1_status in range(2):
				for agent2_status in range(2):
					state = self.encode(task_num, agent1_status, agent2_status)
					for action in range(num_actions):
						#reward = 0
						done = False
						#new_task_num, new_agent1_status, new_agent2_status = task_num, agent1_status, agent2_status
						bin_action = "{0:b}".format(action)
						if len(bin_action) == 1:
							action1 = 0
							action2 = int(bin_action)
						else:
							action1 = int(bin_action[0])
							action2 = int(bin_action[1])

						for new_task_trans in range(self.num_tasks+1):
							#reward = 0
							if agent1_status==0 and agent2_status==0:
								next_state = self.encode(new_task_trans, action1, action2)
								if action1==0 and action2==0 and task_num!=0:
									reward = -8 #major penalty for ignoring
								else:
									reward = self.compute_reward(task_num, action1, action2)
								self.P[state][action].append((1.0 / (self.num_tasks+1), next_state, reward, done))

							elif agent1_status==0 and agent2_status==1:
								reward = -0.25 #for agent2 being busy
								next_state_busy = self.encode(new_task_trans, action1, 1)
								next_state_free = self.encode(new_task_trans, action1, 0)
								if action1==0 and task_num!=0:
									reward = reward - 8 #major penalty for ignoring
								else:
									reward = reward + self.compute_reward(task_num, action1, 0)

								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * BUSY_PROB, next_state_busy, reward, done))
								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * (1.-BUSY_PROB), next_state_free, reward, done))

							elif agent1_status==1 and agent2_status==0:
								reward = -0.25 #for agent1 being busy
								next_state_busy = self.encode(new_task_trans, 1, action2)
								next_state_free = self.encode(new_task_trans, 0, action2)
								if action2==0 and task_num!=0:
									reward = reward - 8 #major penalty for ignoring
								else:
									reward = reward + self.compute_reward(task_num, 0, action2)

								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * BUSY_PROB, next_state_busy, reward, done))
								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * (1.-BUSY_PROB), next_state_free, reward, done))
							elif agent1_status==1 and agent2_status==1:
								reward = -0.5 #both agents busy
								ns_a1_free_a2_free = self.encode(new_task_trans, 0, 0)
								ns_a1_free_a2_busy = self.encode(new_task_trans, 0, 1)
								ns_a1_busy_a2_free = self.encode(new_task_trans, 1, 0)
								ns_a1_busy_a2_busy = self.encode(new_task_trans, 1, 1)

								if task_num!=0:
									reward = reward - 8

								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * (1.-BUSY_PROB) * (1.-BUSY_PROB), ns_a1_free_a2_free, reward, done))
								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * (1.-BUSY_PROB) * BUSY_PROB, ns_a1_free_a2_busy, reward, done))
								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * BUSY_PROB * (1.-BUSY_PROB), ns_a1_busy_a2_free, reward, done))
								self.P[state][action].append(((1.0 / (self.num_tasks+1)) * BUSY_PROB * BUSY_PROB, ns_a1_busy_a2_busy, reward, done))


	def compute_reward(self, task_num, compute_agent1_reward, compute_agent2_reward):
		#bottom left is (0,0) -> location of agent1
		#top right is (1,1) -> location of agent2.
		if task_num == 0:
			return 0

		reward = 0
		if compute_agent1_reward==1:
			reward = reward - DIST_MULT*(self.task_locations[task_num-1, 0]**2 + self.task_locations[task_num-1, 1]**2)**0.5
		if compute_agent2_reward==1:
			reward = reward - DIST_MULT*((1-self.task_locations[task_num-1, 0])**2 + (1-self.task_locations[task_num-1, 1])**2)**0.5
		return reward

	def step(self, actions):
		action1 = actions[0]
		action2 = actions[1]

		action = action1 * 2 + action2

		transitions = self.P[self.state][action]
		i = self._categorical_sample([t[0] for t in transitions])
		p, s, r, d= transitions[i]
		self.laststate = self.state
		self.state = s
		self.lastaction = action
		return (s, r, d, {"prob" : p})

	def set_state(self, task_num):
		self.state = self.encode(task_num, 0, 0)
		self.lastaction = None
		return

	def render(self):
		DISP_SIZE = 500
		display = np.ones((DISP_SIZE, DISP_SIZE, 3))

		if self.lastaction is None:
			#agent1 - bottom left
			cv2.circle(display, (0, DISP_SIZE-1), 25, np.array([0.,0.,0.]), -1)
			#agent2 - top right
			cv2.circle(display, (DISP_SIZE-1, 0), 25, np.array([0.,0.,0.]), -1)
			for i in self.task_locations:
				x = int(i[0] * DISP_SIZE)
				y = int(DISP_SIZE - i[1] * DISP_SIZE)
				#task locations
				cv2.circle(display, (x, y), 10, np.array([0.,0.,1.]), -1)

		else:

			bin_action = "{0:b}".format(self.lastaction)
			if len(bin_action) == 1:
				action1 = 0
				action2 = int(bin_action)
			else:
				action1 = int(bin_action[0])
				action2 = int(bin_action[1])

			task_num, agent1_status, agent2_status = self.decode(self.laststate)
			# print ('Task num: ', task_num)
			#agent1 - bottom left
			if agent1_status == 0:
				cv2.circle(display, (0, DISP_SIZE-1), 25, np.array([0.,0.,0.]), -1)
			else:
				cv2.circle(display, (0, DISP_SIZE-1), 25, np.array([1.,0.,0.]), -1)
			#agent2 - top right
			if agent2_status == 0:
				cv2.circle(display, (DISP_SIZE-1, 0), 25, np.array([0.,0.,0.]), -1)
			else:
				cv2.circle(display, (DISP_SIZE-1, 0), 25, np.array([1.,0.,0.]), -1)

			for cnt, i in enumerate(self.task_locations):
				x = int(i[0] * DISP_SIZE)
				y = int(DISP_SIZE - i[1] * DISP_SIZE)
				#task locations
				if cnt == task_num-1:
					cv2.circle(display, (x, y), 10, np.array([0.,1.,0.]), -1)
				else:
					cv2.circle(display, (x, y), 10, np.array([0.,0.,1.]), -1)

			if task_num != 0:
				#active task location
				x = int(self.task_locations[task_num-1][0] * DISP_SIZE)
				y = int(DISP_SIZE - self.task_locations[task_num-1][1] * DISP_SIZE)
				if agent1_status == 0 and action1 == 1:
					cv2.line(display, (x,y), (0, DISP_SIZE-1), np.array([0.,0.,0.]), 2)
				if agent2_status == 0 and action2 == 1:
					cv2.line(display, (x,y), (DISP_SIZE-1, 0), np.array([0.,0.,0.]), 2)

		cv2.imshow('Display', display)
		cv2.waitKey(30)
		return

	def reset(self):
		self.state = self.encode(np.random.randint(0, self.num_tasks+1), 0, 0)
		self.lastaction = None
		return


	def encode(self, task_num, agent1_status, agent2_status):
		#task_num is the location ID, 0 means no location request
		#agent_status is 0 for Free and 1 for Busy
		# (10+1) 2, 2
		i = task_num
		i = i * 2
		i = i + agent1_status
		i = i * 2
		i = i + agent2_status
		return i


	def encode_decentralized(self, task_num, agent_status):
		#task_num is the location ID, 0 means no location request
		#agent_status is 0 for Free and 1 for Busy
		# (10+1) 2
		i = task_num
		i = i * 2
		i = i + agent_status

		return i

	def decode(self, i):
		out = []
		out.append(i % 2)
		i = i // 2
		out.append(i % 2)
		i = i // 2
		out.append(i)
		assert 0 <= i < self.num_tasks+1
		return reversed(out)

	def _categorical_sample(self, prob_n):
		"""
		Sample from categorical distribution
		Each row specifies class probabilities
		"""
		prob_n = np.asarray(prob_n)
		csprob_n = np.cumsum(prob_n)
		return (csprob_n > np.random.rand()).argmax()

# test = mrt_basic()

# for task_num in range(11):
# 	for agent1_num in range(2):
# 		for agent2_num in range(2):
# 			i = test.encode(task_num, agent1_num, agent2_num)
# 			task_num_rec, agent1_num_rec, agent2_num_rec = test.decode(i)
# 			i_ret = test.encode(task_num_rec, agent1_num_rec, agent2_num_rec)
# 			print ('{} -> {}'.format(i, i_ret))
# 			print ('{} {} {} -> {} {} {}'.format(task_num, agent1_num, agent2_num, task_num_rec, agent1_num_rec, agent2_num_rec))
# 			print()

