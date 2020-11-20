currently ddpg discrete is bugged
which kind of defeats the point of ddpg

of course it cant work for discrete action spaces
idk what i was thinking
theres no gradient going back through action selection
how can it learn at all. 

must use rsample or something?

---------------------------
another day:

you know it should work for discrete action spaces if you are working with the sampleable probabilities