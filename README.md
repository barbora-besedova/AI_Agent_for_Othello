# AI Agent for Othello

This repository contains a project focused on developing an artificial intelligence agent for the game **Othello**. The project was created as part of an Artificial Intelligence and Data Science course during my Erasmus+ exchange stay at **Universitat Politècnica de València**.

The main goal of this project is to build an agent capable of playing Othello by selecting valid and strategic moves based on the current board state.

## Project Overview

Othello is a two-player strategy board game where players take turns placing pieces on the board and flipping the opponent’s pieces. The challenge of this project is to design an AI agent that can evaluate possible actions and choose moves that improve its chance of winning.

The current repository includes the project instructions and a submission template for the final agent.

## Repository Content

```text
Proyecto_Othello_instructions.pdf
agent_submission_template.py
```

### `Proyecto_Othello_instructions.pdf`

This file contains the assignment instructions and project requirements.

### `agent_submission_template.py`

This file provides the required structure for submitting the agent. It includes a `StudentAgent` class and an example Deep Q-Network architecture implemented in PyTorch.

The submitted agent is expected to load a previously trained model and use it only for action selection during evaluation.

## Current Status

At the moment, the repository contains the initial project files and the required agent submission structure. The agent template is prepared for a Deep Q-Learning approach, but the training pipeline and trained model are planned as future additions.

## Planned Work

The next steps of this project are:

* implement the Othello game environment for training,
* create a training pipeline for the AI agent,
* experiment with Q-learning and Deep Q-learning approaches,
* train the agent on different board configurations,
* evaluate the agent against random and heuristic-based opponents,
* improve the neural network architecture,
* tune hyperparameters to improve performance,
* save the trained model as a checkpoint,
* connect the trained model to the final `StudentAgent` class,
* test that the agent always returns legal actions,
* document the training process and final results.

## Planned AI Approach

The main planned approach is to use **Deep Q-Learning**. The agent will learn to estimate the value of possible moves from the current board state. During training, it will improve its strategy by playing games and receiving rewards based on the final outcome.

The final submitted version should not train during evaluation. Instead, it should load a trained PyTorch checkpoint and use the model to select the best legal action.

## Technologies

The project is planned to use:

* Python
* NumPy
* PyTorch
* Reinforcement Learning
* Deep Q-Learning

## Goals

The main goals of this project are:

* to understand how reinforcement learning can be applied to board games,
* to create a functional Othello-playing AI agent,
* to compare simple strategies with a learned strategy,
* to improve the agent’s decision-making over time,
* to prepare a final agent compatible with the required submission format.

## Author

**Barbora Besedová**

Erasmus+ Exchange Student
Universitat Politècnica de València
Artificial Intelligence and Data Science course
