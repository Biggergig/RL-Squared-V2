from openskill.models import PlackettLuce
import pandas as pd


class TournamentSkill:
    def __init__(self):
        self.model = PlackettLuce(balance=True)
        self.bots = {}

    def add_player(self, name):
        # print("adding player", name)
        self.bots[name] = [self.model.rating(name=name)]

    def match(self, name1, name2, goals):
        goals = [float(g) for g in goals]
        for n in [name1, name2]:
            if n not in self.bots:
                # self.add_player(n)
                return
        self.bots[name1], self.bots[name2] = self.model.rate(
            [self.bots[name1], self.bots[name2]],
            scores=[goals[0], goals[2]],
        )

    def getSkill(self, name, elo=False):
        if elo:
            plr = self.bots[name][0]
            return plr.ordinal(alpha=200 / plr.sigma, target=1500)
        return self.bots[name][0]

    def getElos(self):
        elo = {b: self.getSkill(b, elo=True) for b in self.bots}
        return sorted(elo.items(), key=lambda x: x[1])

    def getRanks(self):
        bot_names, bot_ranks = zip(*self.bots.items())
        # print(bot_names, bot_ranks)

        named_ranks = [
            (i, name, p)
            for name, (i, p) in zip(bot_names, self.model.predict_rank(list(bot_ranks)))
        ]
        named_ranks.sort()
        return [(r[1], self.getSkill(r[1], elo=True)) for r in named_ranks]

    def getModelsDF(self, matches):
        df = pd.DataFrame(
            [
                [n, pls.mu, pls.sigma, self.getSkill(n, elo=True), 0, 0, 0]
                for n, (pls,) in self.bots.items()
            ],
            columns=["name", "mean", "var", "elo", "win", "draw", "loss"],
        )
        ranks, rank_prob = zip(
            *self.model.predict_rank([self.bots[n] for n in df["name"]])
        )
        df = (
            df.assign(rank=ranks, rank_prob=rank_prob)
            .set_index("name")
            .sort_values("rank")
        )
        for _, m1, m2, *goals in matches.itertuples():
            if m1 not in self.bots or m2 not in self.bots:
                continue
            df.loc[m1, "win"] += goals[0]
            df.loc[m1, "draw"] += goals[1]
            df.loc[m1, "loss"] += goals[2]
            df.loc[m2, "win"] += goals[2]
            df.loc[m2, "draw"] += goals[1]
            df.loc[m2, "loss"] += goals[0]
        return df
