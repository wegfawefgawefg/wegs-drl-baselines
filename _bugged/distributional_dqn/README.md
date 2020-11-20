Distributional dqn has to be the most complicated DRL algo i've seen.
I understand it at a high level but the "projection onto the support" bit is tough.
I understand that theres aliasing in that projection and why you have a fixed number of atoms.
I understand that theres multiple loss functions you can use for the reward distribution and that 
some are more flawed than others. I know a couple people with strong math backgrounds (math degrees too) who, 
although could understand it, just couldnt implement this.

I feel like this is so complicated to implement it's going to be doomed in its current form.
Maybe reducing the atomized representation down to a normal distribution or a summation of a couple 
high and low frequency components or something would cut out like 80% of the complicated part.

The popularity of PPO, due to its simplicity, really should be a message to future researchers.


On overarchitecting:
In addition, arguably the distribution of rewards should already be represented in the q values in 
an N-Step agent of N atoms lookahead... (More specifically greedy NSTEP will have pathfinded through the distribution, having computed 
the q's for each action at each step ahead)
Maybe we are reaching into the neural network and pulling out an abstraction that should remain abstracted, 
instead of being coerced into the architecture as a prior. Almost assuredly an agent that is real time, or atleast cognissant 
of the passing of time, such as one with a world model that can predict arbitrary seconds ahead, would already 
be predicting and navigating distributions of outcomes. That should come for free with a real time world model.
Requiring less manual labor, and probably not wasting so much compute.