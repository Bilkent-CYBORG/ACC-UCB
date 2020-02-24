"""
This class represents a base arm reward (i.e. quality as stated in the paper)
"""


class Reward:
    def __init__(self, arm, quality):
        self.quality = quality
        self.context = arm.context
        self.arm = arm
