class Stats:
    '''basically a logger'''
    def __init__(self):
        self.num_samples = 0
        self.num_episodes = 0
        self.high_score = 0
        self.scores = []
        self.epsilons = []
        self.last_loss = 0
        
        #   test stats
        self.test_scores = []
        self.high_test_score = 0.0
        self.num_test_episodes = 0

        #   collection stats
        self.num_samples_collected = 0

        #   training stats
        self.num_samples_processed = 0

    def update_training_stats(self, num_samples_processed_inc):
        self.num_samples_processed += num_samples_processed_inc

    def update_test_stats(self, num_test_episodes_inc, latest_test_score):
        self.num_test_episodes += num_test_episodes_inc
        self.test_scores.append(latest_test_score)
        self.high_test_score = max(self.high_test_score, latest_test_score)

    def update_collection_stats(self, num_samples_collected_inc):
        self.num_samples_collected += num_samples_collected_inc 

    def print_test_run_stats(self):
        print(
            (   "num_samples: {}, "
                "num-test-eps: {}, "
                "test-high-score: {:12.3f}, "
                "last-test-score: {:12.3f}, "              
            ).format(
                self.num_samples_processed,
                self.num_test_episodes,
                self.high_test_score,
                self.test_scores[-1],
            )
        )

    def print_episode_end(self):
        print(
            (   "total samples: {}, "
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