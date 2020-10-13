# Evolution Strategies for Board Games

## How to run the code? <br>
For using fully connected layers <br>
`python experiment_runner_modified.py --mutation SM-G-SO --mutation_mag 0.01 --display --domain connect_four  --max_gen 200 gen_length 100 --pop_size 150` <br>

For using convolutional layer <br>
`python experiment_runner_modified.py --mutation SM-G-SO --mutation_mag 0.001 --display --domain connect_four_CNN  --max_gen 200 gen_length 100 --pop_size 150` <br>

For population play <br>
`python experiment_runner_modified.py --mutation SM-G-SO --mutation_mag 0.01 --hidden 256 --layer 1 --population_play --state_archive --max_gen 200 --gen_length 100 --pop_size 150 --fast` <br>

To find about all the arguments use: <br>
`python experiment_runner_modified.py --help` <br>

### TODO:
- Resolve bug in CNN code.
- Add some descriptive figures.

I have also written a blog, which explains what we have done in this project. [Here](https://adityauser.github.io/blog/2020/09/05/PlayingBoardGames.html) is the link to the blog. You are welcome to create issues for your queries regarding the project. <br>

This code is based on Uber-AI safemutation work. You can find the original code [here](https://github.com/uber-research/safemutations). <br>

The second part of my internship was on **Evolution Strategies for Multi-Armed Bandit problem**. Please do checkout [Evolution Strategies for Multi-Armed Bandit problem](https://github.com/adityauser/Evolution-Strategies-RL/tree/master/Multi-Armed%20Bandit).



