# 🧠 Architecture & Mathematical Foundations

This document outlines the four novel contributions of the **FlowMatch-PdM** architecture.



## 1. Bidirectional Mamba Backbone (State-Space Model)
Unlike Diffusion-TS which relies on $O(L^2)$ Attention mechanisms, we utilize a Bidirectional Mamba architecture. This continuous-time differential equation solver processes sequences in $O(L)$ time, capturing both forward degradation trajectories and backward temporal context without the quadratic memory bottleneck, making it highly efficient for long high-frequency vibration windows (e.g., 2048 steps in CWRU).

## 2. Dynamic Conditioned Harmonic Prior
Standard generative models (GANs, DDPMs, standard Flow Matching) map data from a purely stochastic isotropic Gaussian noise distribution $\mathcal{N}(0, I)$. FlowMatch-PdM introduces a physics-informed base distribution. We initialize the Ordinary Differential Equation (ODE) solver with a stochastic harmonic oscillator conditioned on machine operating parameters (e.g., RPM, Load). 
$$p_0(x) = A \cdot \sin(2\pi ft + \phi) + \epsilon$$

## 3. Time-Conditioned Contraction Matching (TCCM)
Machines do not heal. To enforce the physical reality of the Degradation Manifold, we introduce a TCCM penalty. If the generated vector field $v_\theta(t,x)$ implies a positive shift in the Health Index (HI), the loss function severely penalizes the network, constraining the flow strictly to monotonic degradation paths.

## 4. Layer-Adaptive Pruning (LAP)
To prevent the model from wasting compute on stable channels, we adapt the LAP technique from Large Language Models (MoE). Using PyTorch forward hooks, we track the $L_1$ norm of channel activations. Once the variance drops below a stability threshold (the "stable phase"), we apply strict $\alpha$ (individual) and $\beta$ (cumulative) load constraints, dynamically masking up to 30% of the Mamba channels without degrading generative perplexity.