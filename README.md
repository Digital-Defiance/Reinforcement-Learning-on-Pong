
## "Why do we fall, sir? Because it's Reinforcement learning."

Implemented AI player in Pong game, with Custom Environment!

## How does it play?
-> After 140k episodes
<p align="center">
  <img src="DQN/140k/140k_demo.gif" alt="Video">
</p>
  
-> After 220k episodes  
<p align="center">
  <img src="DQN/220k/220k_demo.gif" alt="Video">
</p>

-> After 320k episodes  
<p align="center">
  <img src="DQN/320k/320k_demo.gif" alt="Video">
</p>

## Loss Graph
<p align="center">
  <img src="loss_plot.png" alt="image">
</p>

## Installation

```
git clone https://github.com/a-b-h-a-y-s-h-i-n-d-e/Pong-Game-with-Reinforcement-Learning.git
```
```
pip install -r requirements.txt
```
## Play against AI
```
python main.py 
```


## Train from scratch?

<p align="center">
  <img src="DQN/training_video.gif" alt="Video">
</p>  
-> Delete the trained_model.pth and buffer.pkl files first! <br /> 

```
cd DQN
```
<br />

-> select num_of_episodes in train.py  
```
python train.py
```

