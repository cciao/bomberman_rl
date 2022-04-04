# bomberman_rl
Final Project of Fundamentals of Machine Learning. Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

In this project, we employ handcrafted features to translate game states to vectors and then train a linear regression model to predict the actions. The off-policy multi-step Q-learning algorithm in conjunction with a \epsilon-greedy policy and a prioritized experience replay sampling technique are used. To improve training efficiency, we employ imitation learning, in which we first learn the rule-based agent and subsequently the agent itself.

# master branch
This branch includes our ultimate agent 'expert_rl_w' and the pre-training agent 'base_rl', which is student of rule-based-agent. 

# working branch
The other two agents with different preference are included, 'expert_rl_z' which prefers collecting coins and 'expert_rl_g' which prefers killing opponents.
