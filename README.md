[![CI](https://github.com/Argo-Navis-Lab/scalelab/actions/workflows/ci.yml/badge.svg)](https://github.com/Argo-Navis-Lab/scalelab/actions/workflows/ci.yml)

# **ScaleLab** ? 🤔🧐

**`ScaleLab`** is a measuring instrument for AI training experiments. It answers one big question cheaply: *if a training idea works on a small model, will it still work on a big one?*

**`ScaleLab`** is a small research lab on a **Mac M2-Pro** plus a rented **Tesla T4 GPU**. It answers one question: *When you train a language model with Reinforcement Learning (RL), how much better does it get as you spend more compute, and how does that change with model size?*

Write-ups: [1. The cost model, and what it says about RL's bottlenecks](docs/writeups/01-cost-model.md) (M3).

> [!IMPORTANT]
> **Training Mode.** **ScaleLab** is a research project, built step by step as a hands-on study of how reinforcement learning scales with model size and compute. It follows a written blueprint through eight milestones (M0–M7). M0–M3 are complete; M4, the scaling study and its pre-registered prediction, is in progress.
>
> Every change lands only after passing linting (ruff), strict type checking (pyright) and the full test suite in CI. Until a milestone closes, expect interfaces, configs and file formats to change, and treat results as preliminary unless a write-up reports them.