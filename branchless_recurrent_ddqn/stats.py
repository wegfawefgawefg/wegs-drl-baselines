class Stats:
    def __init__(self):
        self.num_samples = 0
        self.num_episodes = 0
        self.high_score = 0
        self.scores = []
        self.epsilons = []
        self.last_loss = 0

    def print_episode_end(self):
        print(
            ( "total samples: {}, "
                "ep: {}, "
                "high-score: {:12.3f}, "
                "score: {:12.3f}, "
                "epsilon {:12.3f}, "
                "last_loss {:12.3f}"              
            ).format(
                self.num_samples,
                self.num_episodes,
                self.high_score,
                self.scores[-1],
                self.epsilons[-1],
                self.last_loss
            )
        )