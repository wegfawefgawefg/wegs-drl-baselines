class GoRight:
    def __init__(self, frame_limit):
        self.frame = 0
        self.frame_limit = frame_limit

    def step(action):
        state_ = np.array(0.0)
        reward = 0.0
        done = False
        info = None

        if action == 1:
            reward = 1.0

        if self.frame == self.frame_limit:            
            done = True
            return state_, reward, done, info
            
    def reset():
        self.frame = 0