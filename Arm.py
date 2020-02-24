class Arm:
    def __init__(self, unique_id, context, true_mean):
        self.true_mean = true_mean  # Only used by the benchmark
        self.unique_id = unique_id
        self.context = context
