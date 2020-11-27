Actor is a primitive RL algorithm that has no sense of state value.
it just maps states to actions, incentivising actions that end up with good rewards 
in states. 

In the same way that dueling is an attempt to extract a component of reward out into 
the architecture as a prior, value networks themselves were 
extracting the state value out into the architecture. 

There would have been a time before value networks. 
Presumably this is what that algorithm might have looked like.

Performs slightly better than random when hyperparamaters are tuned well.
Primarily avoids dones.
Presumably has no experience about states that do not appear to be immediatly terminal.