"""
Represents a hypercube context region.
"""


class Hypercube:
    def __init__(self, length, center):
        self.center = center
        self.length = length

    def is_pt_in_hypercube(self, point):
        # Translate the hypercube and the point
        for coordinate in (point - self.center):
            if abs(coordinate) > self.length / 2:
                return False
        return True

    def get_dimension(self):
        return len(self.center)

    def __str__(self):
        if len(self.center) == 1:  # for testing purposes
            return "[" + str(self.center[0] - self.length / 2) + " - " + str(self.center[0] + self.length / 2)
        return "center: " + str(self.center) + " length: " + str(self.length)

    def __repr__(self):
        return self.__str__()
