# from elosports, user ddm7018 on github
# modified by Anshuman


class Elo:

    def __init__(self, k):
        self.ratingDict = {}
        self.k = k

    def addPlayer(self, name, rating=1500):
        self.ratingDict[name] = rating

    def gameOver(self, winner, loser, tie=False):
        result = self.expectResult(self.ratingDict[winner], self.ratingDict[loser])
        if tie:
            score = 1 / 2
        else:
            score = 1
        self.ratingDict[winner] += self.k * (score - result)
        self.ratingDict[loser] += self.k * (result - score)

    def expectResult(self, p1, p2):
        exp = (p2 - p1) / 400.0
        return 1 / ((10.0 ** (exp)) + 1)
