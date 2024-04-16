
class Probability:
    prob_informed = None
    prob_engaged = None
    
    prob_vote_engaged = None
    prob_vote_disengaged = None
    prob_vote_correct_informed = None
    prob_vote_correct_uninformed = None

    prob_obj_engaged = None
    prob_obj_disengaged = None
    prob_obj_correct_informed = None
    prob_obj_correct_uninformed = None

    prob_bonus = None

    def __init__(self, prob_vote_engaged, prob_vote_disengaged, prob_vote_correct_informed, prob_vote_correct_uninformed, 
                 prob_obj_engaged, prob_obj_disengaged, prob_obj_correct_informed, prob_obj_correct_uninformed,
                 prob_engaged, prob_informed):
        
        self.prob_vote_engaged = prob_vote_engaged
        self.prob_vote_disengaged = prob_vote_disengaged
        self.prob_vote_correct_informed = prob_vote_correct_informed
        self.prob_vote_correct_uninformed = prob_vote_correct_uninformed

        self.prob_obj_engaged = prob_obj_engaged
        self.prob_obj_disengaged = prob_obj_disengaged
        self.prob_obj_correct_informed = prob_obj_correct_informed
        self.prob_obj_correct_uninformed = prob_obj_correct_uninformed

        self.prob_engaged = prob_engaged
        self.prob_informed = prob_informed

        self.prob_bonus = 0

    @staticmethod
    def will_vote(self, voter, prob_vote):
        return (voter.is_engaged() and self.prob_vote_engaged > prob_vote) or \
                (not voter.is_engaged() and self.prob_vote_disengaged > prob_vote)

    @staticmethod
    def will_vote_correct(self, voter, prob_vote_correct):
        return (voter.is_informed() and (self.prob_vote_correct_informed + self.prob_bonus) > prob_vote_correct) or \
                (not voter.is_informed() and (self.prob_vote_correct_uninformed + self.prob_bonus) > prob_vote_correct)
    
    @staticmethod
    def will_object(self, voter, prob_obj):
        return (voter.is_engaged() and self.prob_obj_engaged > prob_obj) or \
                (not voter.is_engaged() and self.prob_obj_disengaged > prob_obj)
    
    @staticmethod
    def will_object_correct(self, voter, prob_obj_correct):
        return (voter.is_informed() and self.prob_obj_correct_informed > prob_obj_correct) or \
                (not voter.is_informed() and self.prob_obj_correct_uninformed > prob_obj_correct)

    @staticmethod
    def objection_effect(self, effect):
        if effect:
            self.prob_bonus = 0.2
        else:
            self.prob_bonus = -0.2
        