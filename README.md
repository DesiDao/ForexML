# ForexML

## Overview

This project implements a Reinforcement Learning (RL) Forex Trading Bot, supported by supervised learning models for price movement classification and LSTM-based price prediction. The bot leverages deep learning techniques to make informed trading decisions based on historical and real-time Forex market data.

## Project Components

### 1. Forex Price Movement Classification

A supervised learning model that classifies whether a currency pair's price will move up, down, or stay neutral over a specific time frame.

Features include technical indicators such as Moving Averages, RSI, MACD, and Bollinger Bands.

Model: Random Forest, Gradient Boosting, or Neural Networks.

### 2. LSTM-Based Forex Price Prediction

A deep learning model that uses Long Short-Term Memory (LSTM) networks to forecast future Forex prices.

The model is trained on historical price data to identify long-term trends and short-term fluctuations.

Enhances the RL agent by providing a learned forecast for decision-making.

### 3. Reinforcement Learning Forex Trading Bot

Uses Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), or Advantage Actor-Critic (A2C) to learn optimal trading strategies.

The bot's state representation includes outputs from the classification and LSTM models, along with real-time market data.

Rewards are based on profit/loss per trade, encouraging the agent to develop an efficient strategy.

## Technologies Used

Programming Language: Python

Machine Learning Libraries: TensorFlow, PyTorch, Scikit-learn

Data Processing: Pandas, NumPy

Market Data API: Alpha Vantage, OANDA, Yahoo Finance

Visualization: Matplotlib, Seaborn

## Installation & Setup

### Clone the repository:

> git clone https://github.com/your-username/forex-rl-bot.git
> cd forex-rl-bot

### Install dependencies:

> pip install -r requirements.txt

Configure market data API keys in a .env file.

### Run the training scripts for each model before deploying the RL bot:

> python train_classification.py
> python train_lstm.py
> python train_rl_bot.py

## Results & Performance Evaluation

The project includes backtesting on historical data to evaluate model performance.

Performance metrics such as Sharpe Ratio, Win Rate, and Maximum Drawdown are tracked.

The RL bot's efficiency is compared against baseline strategies (Buy & Hold, Moving Average Crossover, etc.).

## Future Improvements

Implementing transformer-based models (e.g., GPT, BERT) for sentiment analysis.

Adding economic news impact analysis using NLP.

Optimizing the RL agent with multi-agent reinforcement learning (MARL).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussions.
