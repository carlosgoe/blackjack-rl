# blackjack-rl

A Python implementation of Blackjack as a reinforcement learning environment. Possible actions are stick (S), hit (H), double (D), and surrender (R).

A deep Q learning algorithm (implemented with TensorFlow and Keras) is used to train an agent to maximize rewards in the game. The neural network configuration can be exported as a .h5 file.

The agent's policy is converted to a strategy table that is tested/evaluated across multiple iterations and compared to the basic strategy. 

The deep Q network includes code snippets from Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
