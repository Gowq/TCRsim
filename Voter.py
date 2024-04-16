import random


def get_inform(p_informed):
    return random.random() < p_informed


def get_engagement(p_engaged):
    return random.random() < p_engaged


class Voter():
    def __init__(self, tokens, p_engaged, p_informed):
        self.__mTokens = tokens
        self.__mEngaged = get_engagement(p_engaged)
        self.__mInformed = get_inform(p_informed)
        self.__mVote = None

    def is_engaged(self):
        return self.__mEngaged

    def is_informed(self):
        return self.__mInformed

    def get_tokens(self):
        return self.__mTokens

    def set_tokens(self, tokens):
        if tokens <= 0.0:
            self.__mTokens = 0.0
        else:
            self.__mTokens = tokens

    def set_vote(self, vote):
        self.__mVote = vote

    def get_vote(self):
        return self.__mVote

    def has_enough_tokens(self, cost):
        return self.get_tokens() > cost
