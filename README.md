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

## Some of the training and gameplay output for illustration:
```
Neural Net was initialized with the following values
#input nodes: 2 #hidden nodes: 4 #outputs: 2
Neural Net was initialized with the following values
#input nodes: 2 #hidden nodes: 4 #outputs: 2
Training Neural Network 2
Neural Net was initialized with the following values
#input nodes: 2 #hidden nodes: 4 #outputs: 2
Training Neural Network 3
0 :cards: P: ['7', 'Q'] D: K NN predicted: Hold conf: 0.12 desired= Draw right %=
0.0
1 :cards: P: ['7', 'J'] D: K NN predicted: Hold conf: 0.12 desired= Draw right %=
0.0
2 :cards: P: ['10', '8'] D: 8 NN predicted: Hold conf: 0.12 desired= Hold right %=
33.333333333333336
3 :cards: P: ['J', '10'] D: 9 NN predicted: Hold conf: 0.12 desired= Hold right %=
50.0
4 :cards: P: ['9', '8'] D: J NN predicted: Hold conf: 0.12 desired= Draw right %=
40.0
5 :cards: P: ['J', '2'] D: 8 NN predicted: Hold conf: 0.12 desired= Draw right %=
33.333333333333336
10000 :cards: P: ['K', 'Q'] D: 8 NN predicted: Hold conf: 0.60 desired= Hold right#== 66.43335666433357
Draw a 3th Card
30000 :(NN2)cards: P: ['7', '9', '6'] D: K NN predicted: Hold conf: 1.00 desired=
Hold right % (NN2)= 84.13944150392408
40000 :cards: P: ['2', '7'] D: 4 NN predicted: Draw conf: 0.84 desired= Draw right
%= 80.13299667508312
Draw a 3th Card
40000 :(NN2)cards: P: ['2', '7', '10'] D: 4 NN predicted: Hold conf: 0.97 desired=
Hold right % (NN2)= 83.92493552327949
50000 :cards: P: ['Q', 'K'] D: 9 NN predicted: Hold conf: 0.79 desired= Hold right
%= 80.92638147237055
60000 :cards: P: ['Q', '3'] D: 6 NN predicted: Draw conf: 0.75 desired= Draw right
%= 81.40697655039082
Draw a 3th Card
60000 :(NN2)cards: P: ['Q', '3', 'Q'] D: 6 NN predicted: Hold conf: 1.00 desired=
Hold right % (NN2)= 83.76630691931595
70000 :cards: P: ['K', '4'] D: 7 NN predicted: Draw conf: 0.62 desired= Draw right
%= 81.73740375137498
Draw a 3th Card
70000 :(NN2)cards: P: ['K', '4', '2'] D: 7 NN predicted: Hold conf: 0.43 desired=
Draw right % (NN2)= 83.80893658837357
80000 :cards: P: ['3', 'K'] D: A NN predicted: Draw conf: 0.61 desired= Draw right
%= 82.01522480968988
Draw a 3th Card
80000 :(NN2)cards: P: ['3', 'K', 'Q'] D: A NN predicted: Hold conf: 1.00 desired=
Hold right % (NN2)= 83.76632140085698
90000 :cards: P: ['A', 'J'] D: 8 NN predicted: Draw conf: 0.85 desired= Draw right
%= 82.21241986200154
Draw a 3th Card
90000 :(NN2)cards: P: ['A', 'J', '2'] D: 8 NN predicted: Draw conf: 0.76 desired=
Draw right % (NN2)= 83.73909530293734
Draw a 4th Card
90000 :(NN3)cards: P: ['A', 'J', '2', 'A'] D: 8 NN predicted: Draw conf: 0.22
desired= Draw right % (NN3)= 81.82065661240267
Neural Networks have been trained, these are their accuracy rates:
NN1: 82.352
NN2: 83.78007242962344
NN3: 81.70767571806685
```
## Gameplay example:
```
Lets play a couple games to test them out:
Game# 0 : Player won!
Game# 0 : Player Cards ['6', '3', '2', '6'] Total: 17
Game# 0 : Dealer Cards ['K', 'A', 'A', '10'] Total: 22
Game# 1 : Player won!
Game# 1 : Player Cards ['10', '3', '6'] Total: 19
Game# 1 : Dealer Cards ['J', '8'] Total: 18
Game# 2 : Player won!
Game# 2 : Player Cards ['Q', '3', '7'] Total: 20
Game# 2 : Dealer Cards ['A', '10', 'A', '3', '3'] Total: 18
Game# 3 : Player lost :(
Game# 3 : Player Cards ['8', 'J'] Total: 18
Game# 3 : Dealer Cards ['A', '4', 'K', '3'] Total: 18
Game# 4 : Player won!
Game# 4 : Player Cards ['Q', 'J'] Total: 20
Game# 4 : Dealer Cards ['7', '9', 'K'] Total: 26
Game# 5 : Player Won :(
Game# 5 : Player Cards ['6', 'Q', '5'] Total: 21
Game# 5 : Dealer Cards ['9', 0] Total: 9
Game# 6 : Player lost :(
Game# 6 : Player Cards ['10', '4', '4'] Total: 18
Game# 6 : Dealer Cards ['9', 'J'] Total: 19
Game# 7 : Player lost :(
Game# 7 : Player Cards ['7', 'Q'] Total: 17
Game# 7 : Dealer Cards ['A', '9', '10'] Total: 20
Game# 8 : Player lost :(
Game# 8 : Player Cards ['10', '4', '3'] Total: 17
Game# 8 : Dealer Cards ['7', 'A', '3', '8'] Total: 19
Game# 9 : Player won!
Game# 9 : Player Cards ['J', '10'] Total: 20
Game# 9 : Dealer Cards ['2', 'A', '2', 'A', '8', '3'] Total: 17
Game Stats
Total games: 10
Total wins: 6
Total losses: 4
```
