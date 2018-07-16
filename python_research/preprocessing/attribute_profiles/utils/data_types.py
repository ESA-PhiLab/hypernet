
class Pixel:

    def __init__(self, x, y, gray_level):
        self.x = x
        self.y = y
        self.gray_level = gray_level

    @property
    def coords(self):
        return self.y, self.x

    def __eq__(self, other):
        if type(other) != Pixel:
            return False
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if type(other) != Pixel:
            return self.gray_level < other
        else:
            return self.gray_level < other.gray_level
