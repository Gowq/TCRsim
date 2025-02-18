import random


class Item():
    PERCENT_CHANCE = 0.5

    def __init__(self):
        self.__mValid = None
        self.__mAccepted = None
        self.__mObjection = False

    def set_validity(self, valid=None):
        if (valid == None):
            self.__mValid = random.random() < Item.PERCENT_CHANCE
        else:
             self.__mValid = valid

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

