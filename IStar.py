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

    def to_list(self):
        return [self.star_name, self.x, self.y, self.magnitude, self.counts]

    def __str__(self):
        return (str(self.star_name) + ", " + str(self.x) + ", " + str(
            self.y) + ", " + str(self.magnitude) + ", " + str(self.counts))
