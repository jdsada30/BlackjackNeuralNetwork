# Blackjack Neural Network Implementation

This implementation of BlackJack uses 3 Neural Networks. (1st NN decided whether player should draw a 3nd card, 2nd NN decided whether player should draw a 4rd card, 3rd NN decided whether player should draw a 5th card)
It was implemented in Python and uses NumPy Library to handle Matrix math for feedforward and backpropagation training algorithm.

## Files in Repository
blackJackNN.py: Main File to run, will create and train the neural networks and at
the end call the GamePlay.py simulation to test them by playing some games. <br />
Simulation.py: Used to train neural network by playing simulations of draw and
hold possibilities <br />
GamePlay.py: Simulates blackjack games given the trained neural networks and
calculates number of wins and losses with the trained Neural Netoworks used as
inputs to decided when to hold/draw throughout the game. <br />
