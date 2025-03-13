from openskill.models import PlackettLuce


class TournamentSkill:
    def __init__(self):
        self.model = PlackettLuce()
        self.bots = {}

    def add_player(self, name):
        self.bots[name] = [self.model.rating(name=name)]

    def match(self, model1, model2, goals):
        name1, name2 = model1.name, model2.name
        goals = [float(g) for g in goals]
        for n in [name1, name2]:
            if n not in self.bots:
                self.add_player(n)
        self.bots[name1], self.bots[name2] = self.model.rate(
            [self.bots[name1], self.bots[name2]], weights=[[goals[0]], [goals[1]]]
        )

    def getSkill(self, name):
        return self.bots[name][0]

    def ranks(self):
        return [
            *zip(
                list(self.bots.values()), self.model.predict_rank([*self.bots.values()])
            )
        ]
