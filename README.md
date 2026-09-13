# Reinforcement Learning: Sim2Real Transfer on Hopper-v4

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-brightgreen)

## Overview
This project explores the fundamental of Deep Reinforcement Learning (RL) applied to continuous control tasks, focusing on the highly unstable `Hopper-v4` environment.

To ensure enterprise-level code quality, the architecture is modular: Neural Network (`policy.py`), RL mathematical engines (`agent.py`), and the Training CLI (`train.py`) are decoupled using OOP and Strategy/Factory patterns. Experiments configurations are fully managed via `.yaml` files.

## Part 1: Policy Gradient Methods (REINFORCE vs. Actor-Critic)

The first phase involved implementing two foundational Policy Gradient algorithms from scratch to understand the mathematical mechanics, and practical limitations, of continuous robotic control.

### REINFORCE

REINFORCE relies of Monte Carlo returns, meaning the agent evaluates an action based on the cumulative reward of the entire episode.

* **Gradient Explosion & Statistical Normalization:** Wihtout return normalization, Vanilla REINFORCE exhibits unbounded grandient variance. As a episode lenghts increse, the Actor Loss scales disproportionally (frequently exceeding $100k$). By statistically normalizing the episodic returns, we creates a dynamic baseline, successfully bounding the loss ($\approx [-50, 50]$) and stabilizing the gradients. However, this stabilitization induces **Risk Aversion**: the agent converges to a safe, sub-optimal local minimum (plateauing at $\approx 300$ reward).
* **Extreme Seed Sensitivity:** Despite normalization, multi-seed evaluations revealed massive algorithmic variance. Because Monte Carlo returns sum all rewards (good and bad) across an episode, the optimization path is highly susceptible to initial conditions and environmental stochasticity. This results in wide confidence intervals, making the algorithm's performance highly unpredictable.
* **The Temporal Credit Assignment Problem:** The core limitation of REINFORCE is its inability to assign credit to specific state-action pairs. An optimal propulsive action at step 10, followed by a mechanical failure at step 400, results in the entire trajectory being penalized, fundamentally hindering sample efficiency and convergence.

![](assets/plots/reinforce_ablation.png)
*Figure 1: Ablation study demonstrating unbounded gradient variance in Vanilla REINFORCE and stability induced by normalization.*

### Actor-Critic

To address the credit assignment problem, a Value Fcuntion (the Critic) is introduced to compute the 1-step Temporal Difference (TD) Error (Advantage), shifting from high-variance global returns to low-variance local evaluations.

While Actor-Critic theoretically mitigates the high variance of REINFORCE, empirical testing revealed the **optimization fragility** of foundational on-policy methods in continuous domains.

* **Variance Reduction vs. Premature Convergence:** Actor-Critic successfully drastically reduces the optimization variance in the early-to-mid stages of training (demonstrated by a much narrower standard deviation compared to REINFORCE). However, this stability induces **Risk Aversion**. The agent prematurely converges to a sub-optimal local minimum (plateauing at $\approx 200$ reward), learning a passive balancing strategy to survive rather than exploring forward locomotion.
* **Catastrophic Policy Collapsing:** Because Vanilla Actor-Critic updates the policy *online* without constraints (e.g., Trust Regions), taking a gradient step based on a poorly estimated Advantage can instantly corrupt the Actor's weights. We observed late-stage training instability where the policy forgets its learned behavior, causing the reward to drop precipitously.
* **The Ascending $\sigma$ Anomaly:** By continuously logging the mean standard deviation ($\sigma$) of the Actor's action distribution, we observed a counter-intuitive phenomenon during policy degradations. Instead of monotonic exploration decay (where $\sigma \to 0$ as the policy becomes confident), $\sigma$ frequently spiked back to high values. Mathematically, this acts as a network compensation mechanism: once the learned action mean ($\mu$) is corrupted and leads to failure, the optimization process drastically inflates the exploration noise ($\sigma$) in a blind attempt to escape the deteriorating policy space.

![](assets/plots/ac_vs_reinforce.png)
*Figure 2: Multi-seed evaluation demonstrating the variance reduction of Actor-Critic, although also its premature convergence and the extreme high variance of REINFORCE (even with normalization).*

### Qualitative Results (Evaluation on Trained Agents)

| REINFORCE (Normalized) | Actor-Critic |
|------------------------|--------------|
| <img src="assets/gifs/reinforce_norm_42.gif" width="400"/> | <img src="assets/gifs/ac_s42.gif" width="400"/> |

* **REINFORCE Best-Checkpoint Bias:** Due to high variance, the saved "best model" represents a lucky outlier trajectory.
* **Actor-Critic Stable but Risk-Averse:** Converges to a safe local minimum. The agent balances perfectly on the spot to avoid falling, but refuses to move forward.