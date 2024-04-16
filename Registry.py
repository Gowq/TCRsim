
class Registry:
    num_items = None
    num_voters = None
    default_tokens = None
    stake = None
    initial_stake = None
    delta = None

    stake_pool = None
    total_tokens = None
    code_complexity = None
    time = None
    blocked = None

    def __init__(self, num_voters, default_tokens, stake, initial_stake, delta, code_complexity):
        self.num_items = 1
        self.num_voters = num_voters
        self.default_tokens = default_tokens
        self.stake = stake
        self.initial_stake = initial_stake
        self.delta = delta
        self.stake_pool = 0
        self.total_tokens = num_voters * default_tokens
        self.code_complexity = code_complexity
        self.time = code_complexity
        self.blocked = False


    def increase_itens(self):
        self.num_items += 1

    def block(self):
        self.blocked = True
    
    def unblock(self):
        self.blocked = False