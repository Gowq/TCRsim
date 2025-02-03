import random

def set_rand_inform(p_informed: float) -> bool:
    """Randomly determine if voter is informed based on probability"""
    if not 0 <= p_informed <= 1:
        raise ValueError(f"p_informed must be between 0-1. Got {p_informed}")
    return random.random() < p_informed

def set_rand_engagement(p_engaged: float) -> bool:
    """Randomly determine if voter is engaged based on probability"""
    if not 0 <= p_engaged <= 1:
        raise ValueError(f"p_engaged must be between 0-1. Got {p_engaged}")
    return random.random() < p_engaged

class Voter:
    def __init__(self, 
                 tokens: float,
                 p_engaged: float = None,  
                 p_informed: float = None, 
                 voter_id: int = None,
                 is_engaged: bool = None,
                 is_informed: bool = None):

        self.__mTokens = tokens
        self.voter_id = voter_id

        if (p_engaged and is_engaged == None):
            print("p_engaged and is_engaged cannot be both None")

        if (p_informed and is_informed == None):
            print("p_informed and is_informed cannot be both None")
        
        # Set engagement status
        if is_engaged is None:
            self.__mEngaged = set_rand_engagement(p_engaged)
        else:
            self.__mEngaged = is_engaged
            
        # Set information status
        if is_informed is None:
            self.__mInformed = set_rand_inform(p_informed)
        else:
            self.__mInformed = is_informed

        self.__mVote = None

    # Keep existing getter/setter methods
    def get_engagement(self) -> bool: return self.__mEngaged
    def get_information(self) -> bool: return self.__mInformed
    def get_tokens(self) -> float: return self.__mTokens
    def set_tokens(self, tokens: float): self.__mTokens = max(0.0, tokens)
    def set_vote(self, vote: bool): self.__mVote = vote
    def get_vote(self) -> bool: return self.__mVote
    def has_enough_tokens(self, cost: float) -> bool: return self.__mTokens >= cost
    def set_inform(self, informed: bool): self.__mInformed = informed
    def set_engaged(self, engaged: bool): self.__mEngaged = engaged