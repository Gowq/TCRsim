import random


class Item():
    PERCENT_CHANCE = 0.85

    def __init__(self):
        self.__mValid = self.set_validity()
        self.__mAccepted = None
        self.__mObjection = False

    @staticmethod
    def set_validity():
        return random.random() < 0.85

    def is_valid(self):
        return self.__mValid

    def set_acceptance(self, accepted):
        self.__mAccepted = accepted

    def is_accepted(self):
        return self.__mAccepted

    def is_objection(self):
        return self.__mObjection

    def set_objection(self):
        self.__mObjection = True

    def remove_objection(self):
        self.__mObjection = False
