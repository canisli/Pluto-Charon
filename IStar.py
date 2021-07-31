class IStar:
    """
    Each instance of IStar describes a single star
    """

    def __init__(self, star_name=None, x=None, y=None, magnitude=None, counts=None, table_row=None):
        if table_row is None:
            self.star_name = star_name
            self.x = x
            self.y = y
            self.magnitude = magnitude
            self.counts = counts
        else:
            self.star_name = table_row['name']
            self.x = float(table_row['x'])
            self.y = float(table_row['y'])
            try:
                self.magnitude = float(table_row['mag'])
            except ValueError:
                self.magnitude = table_row['mag']
            self.counts = float(table_row['counts'])

    def to_list(self):
        return [self.star_name, self.x, self.y, self.magnitude, self.counts]

    def __str__(self):
        return (str(self.star_name) + ", " + str(self.x) + ", " + str(
            self.y) + ", " + str(self.magnitude) + ", " + str(self.counts))
