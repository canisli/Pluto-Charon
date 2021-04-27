#!/usr/bin/python3


class IStar:
    """
    Each instance of IStar describes a single star
    """

    def __init__(self, star_name, x, y, magnitude, counts):
        self.star_name = star_name
        self.x = x
        self.y = y
        self.magnitude = magnitude
        self.counts = counts

    def __str__(self):
        return (str(self.star_name) + " located at x=" + str(self.x) + " and y=" + str(
            self.y) + " with magnitude " + str(self.magnitude) + " and counts " + str(self.counts))
