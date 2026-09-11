# Reinforcement Learning: Sim2Real Transfer on Hopper-v4

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-brightgreen)

## Overview
This project explores the fundamental of Deep Reinforcement Learning (RL) applied to continuous control tasks, focusing on the highly unstable `Hopper-v4` environment.

To ensure enterprise-level code quality, the architecture is modular: Neural Network (`policy.py`), RL mathematical engines (`agent.py`), and the Training CLI (`train.py`) are decoupled using OOP and Strategy/Factory patterns. Experiments configurations are fully managed via `.yaml` files ensuring reproducibility.

## Part 1: Policy Gradient Methods (REINFORCE vs. Actor-Critic)

The first phase required to implement two foundational Policy Gradient algorithms from scratch to understand the mathematical mechanics, and practical limitations, of continuous robotic control.

### REINFORCE

REINFORCE relies of Monte Carlo returns, meaning the agent evaluates an action based on the cumulative reward of the entire episode.

* **Vanilla & Static Baseline ($b=20$):** Without normalization, the Actor Loss explodes (reaching values $> 100k$) as episode lenghts increase. A static baseline proved helpful only in the very early epochs but became numerically irrelevant as the agent's reward scaled from 15 to 300+.
* **Statistical Normalization:** By normalizing the episodic returns, the baseline becomes dynamic. This successfully bounded the loss and stabilized the gradients. However, the agent struggled to surpass the $\approx 300$ reward threshold.
* **The Credit Assignment Problem:** The major limitation of REINFORCE is its inability to assign credit to specific actions. A *brilliant jump* at step 10 followed by a *fatal fail* at step 400 results in the entire trajectory being penalized. This forces the agent into a **sub-optimal survival policy** (e.g., awkwardly balancing on the spot instead of running forward).

### Actor-Critic

To solve the credit assignment problem, a Value Fcuntion (the Critic) is introduced to compute the 1-step Temporal Difference (TD) Error.

While Actor-Critic theoretically solves the high variance of REINFORCE, empirical testing revealed the **harsh reality of applied Deep RL**: what is theoretically superior is not always pratically stable.

#### 1. The Seed Sensitivity Problem

Academic literature often presents Vanilla Actor-Critic as a robust baseline, usually omitting the exhaustive hyperparameter tuning (sweeping hundreds of combinantions of Learning Rates, architectures, and initialiation) required to make it work. In our test, the algorithm exhibited extreme sensitivity to random seed. For instance, `seed: 123` sucessfully reached rewards of $\approx 1000$, while `seed: 42` collapsed early on.

<!-- TODO: maybe add some plots about seed sensitivity -->

#### 2. Catastrophic Policy Collapse

Since Vanilla Actor-Critic updates the policy *online*, a single poorly estimated Advantage from the Critic can yield a destructive gradient update. This instantly corrupts the Actor's weights, completely erasing thousands of epochs of progress in a single step (Catastrophic Forgetting).

#### 3. The Ascending $\sigma$ Anomaly

By tracking the mean standard deviation ($\sigma$) of the Actor's action distribution, we observed that instead of decaying as the agent becomes confident, $\sigma$ frequently spiked up to values $> 1.0$ during a policy collapse. Mathematically, this is the neural network entering a "panic state": once the learned mean ($\mu$) is corrupted and leads to negative rewards, the policy drastically inflates its exploration noise ($\sigma$) in a desperate, blind attempt to find a new safe state.

### Qualitative Results (Trained Agents)

<!-- TODO -->