import random
import csv
import matplotlib.pyplot as plt
from copy import copy
from Voter import Voter
from Item import Item
from TCR import TCR
from Probability import Probability
from Registry import Registry
import sys
sys.stdout.reconfigure(encoding='utf-8')


class Simulation:
    item_validity = None
    valid_outcome = 0

    def __init__(self, probability, registry, quantity_votes, debug=False):
        self.probability = probability
        self.registry = registry
        self.quantity_votes = quantity_votes
        self.vote_results = []  
        self.item_array = []
        self.tcr_array = []
        self.debug = debug 
        self.next_voter_id = 0
        self.round_number = 0
        self.voters = []  # Lista de eleitores personalizados
        self.registry.num_voters = 0  # Reinicia o contador de eleitores

    def process_voter(self, voter, item):
        prob_object = random.random()
        prob_vote = random.random()

        if self.probability.will_object(voter, prob_object) and not self.registry.blocked:
            return self.process_objection(voter, item)
        
        if self.probability.will_vote(voter, prob_vote):
            return self.process_vote(voter, item)
        
        return voter

    def process_vote(self, voter, item):
        if not voter.has_enough_tokens(self.registry.stake):
            return voter

        old_tokens = voter.get_tokens()
        prob_correct = random.random()
        status = self.get_voter_status(voter)
    
        self.registry.stake_pool += self.registry.stake
        voter.set_tokens(old_tokens - self.registry.stake) 

        if self.debug:
            print(f"[Round {self.round_number}] Voter {voter.voter_id} ({status}) "
                  f"perdeu {self.registry.stake} tokens (Aposta). "
                  f"Tokens: {old_tokens} → {voter.get_tokens()}")
       
        if self.probability.will_vote_correct(voter, prob_correct):
            new_vote = item.is_valid()
        elif self.registry.blocked:
            new_vote = not item.is_valid()
        else:
            return self.process_objection(voter, item)
        
        old_vote = voter.get_vote()  
        voter.set_vote(new_vote)

        if self.debug and old_vote != new_vote:
            print(f"[Round {self.round_number}] Voter {voter.voter_id} ({status}) "
                  f"votou: {new_vote}")

        return voter

    def process_objection(self, voter, item):
        objection_cost = 2 * self.registry.stake
        if not voter.has_enough_tokens(objection_cost):
            return voter

        old_tokens = voter.get_tokens()
        status = self.get_voter_status(voter)
        prob_correct = random.random()
        self.registry.stake_pool += objection_cost
        voter.set_tokens(old_tokens - objection_cost)

        if self.debug:
            print(f"[Round {self.round_number}] Voter {voter.voter_id} ({status}) "
                  f"perdeu {objection_cost} tokens (Objeção). "
                  f"Tokens: {old_tokens} → {voter.get_tokens()}")
            print(f"[Round {self.round_number}] Voter {voter.voter_id} ({status}) "
                  f"fez uma objeção.")
            
        voter.set_vote(False)  
        self.registry.increase_itens()
        item.set_objection()

        if self.debug:  
            print(f"[Debug] Voter {voter.voter_id} fez uma objeção.")

        if self.probability.will_object_correct(voter, prob_correct):
            self.probability.objection_effect(True)
        else:
            self.probability.objection_effect(False)
        
        self.registry.block()
        return voter
    

        

    def run_simulation_round(self, round_time):
        self.round_number += 1
        self.registry.stake_pool = 0

        if self.debug:
            print(f"\n{'='*40}")
            print(f"Round {self.round_number} - Início")
            print(f"{'='*40}")

        votes_round = min(
            int(random.gauss(self.quantity_votes * round_time, 1)),
            self.registry.num_voters
        )

        current_voters = []
        item = Item()

        if (round_time == self.registry.code_complexity):
            item.set_validity()
            self.item_validity = item.is_valid()
        else:
            item.set_validity(self.item_validity)

        # Embaralha a ordem dos eleitores para evitar viés
        shuffled_voters = self.voters.copy()
        random.shuffle(shuffled_voters)

        for j in range(votes_round):
            # Como votes_round == número total de eleitores, podemos iterar na ordem embaralhada
            voter = copy(shuffled_voters[j % len(shuffled_voters)])
            voter.set_vote(None)
            processed_voter = self.process_voter(voter, item)
            current_voters.append(processed_voter)

        # Atualiza a lista principal de eleitores – a ordem aqui pode ser mantida aleatória ou redefinida
        self.voters = current_voters.copy()

        accepted = sum(v.get_vote() for v in current_voters if v.get_vote() is not None)
        rejected = sum(1 for v in current_voters if v.get_vote() is False)
        item.set_acceptance(accepted >= rejected)

        if self.debug:
            print(f"\n{'='*40}")
            print(f"Round {self.round_number} - Resultado Final")
            print(f"Item válido: {item.is_valid()} | Aceito: {accepted >= rejected}")
            print(f"{'='*40}\n")

        self.item_array.append(item)
        self.vote_results.append(current_voters)
        num_winners = sum(1 for v in current_voters if v.get_vote() == item.is_accepted())
        self.update_tokens(current_voters, num_winners)
        self.update_tcr()


    def update_tokens(self, voters, participants):
        if participants == 0:
            return

        # Distribuição normal dos tokens apostados
        stake_total = self.registry.stake_pool
        winners = sum(1 for v in voters if v.get_vote() == self.item_array[-1].is_accepted())
        
        if winners > 0:
            stake_per_voter = stake_total / winners
            remainder = stake_total - (stake_per_voter * winners)

            for idx, voter in enumerate(voters):
                if voter.get_vote() == self.item_array[-1].is_accepted():
                    add_amount = stake_per_voter + (remainder if idx == 0 else 0)
                    voter.set_tokens(voter.get_tokens() + add_amount)
                    
                    if self.debug:
                        status = self.get_voter_status(voter)
                        print(f"[Round {self.round_number}] Voter {voter.voter_id} ({status}) "
                            f"ganhou {add_amount:.2f} tokens.")

        # Mecanismo de inflação apenas para vencedores
        if self.registry.delta > 0:
            # Calcula inflação baseada no total atual de tokens
            inflation_amount = self.registry.total_tokens * self.registry.delta
            
            # Filtra apenas os vencedores
            current_winners = [v for v in voters if v.get_vote() == self.item_array[-1].is_accepted()]
            
            if current_winners:
                # Calcula a participação proporcional de cada vencedor
                total_winner_tokens = sum(winner.get_tokens() for winner in current_winners)
                
                if total_winner_tokens > 0:
                    for winner in current_winners:
                        share = winner.get_tokens() / total_winner_tokens
                        winner.set_tokens(winner.get_tokens() + (share * inflation_amount))
                    
                    # Atualiza o total de tokens com a inflação
                    self.registry.total_tokens += inflation_amount

            # Ajusta o stake usando o valor inicial ajustado pela inflação
            self.registry.stake = self.registry.initial_stake * (1 + self.registry.delta) ** self.round_number
            
            if self.debug:
                print(f"\n[Inflação] Stake ajustado para: {self.registry.stake:.2f}")
                print(f"[Inflação] Total de tokens após ajuste: {self.registry.total_tokens:.2f}")

    def update_tcr(self):
        # Use registry's total_tokens that includes inflation
        new_tcr = TCR(self.registry.total_tokens)
        
        if self.item_array[-1].is_valid() and self.item_array[-1].is_accepted():
            new_tcr.set_tcr_value(1, 0, 0, 0)
        elif self.item_array[-1].is_valid() and not self.item_array[-1].is_accepted():
            new_tcr.set_tcr_value(0, 0, 0, 1)
        elif not self.item_array[-1].is_valid() and self.item_array[-1].is_accepted():
            new_tcr.set_tcr_value(0, 0, 1, 0)
        else:
            new_tcr.set_tcr_value(0, 1, 0, 0)
        
        self.tcr_array.append(new_tcr)

    def run(self):
        round_time = self.registry.code_complexity
        
        while round_time > 1:

            self.registry.unblock()
            self.run_simulation_round(round_time)            
            round_time /= 2

            if (self.item_array[-1].is_accepted() and self.registry.blocked):
                continue
            else: 
                break

        self.write_csv()
        self.generate_plots()

        if (self.item_array[-1].is_valid() == self.item_array[-1].is_accepted()):
            self.valid_outcome += 1

        #print(self.valid_outcome)
        #print("valid outcome: ", (self.item_array[-1].is_valid() == self.item_array[-1].is_accepted()))

    def write_csv(self):
        with open("output.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["TCR Val", "Valid", "Accept"] + [f"V{i+1}" for i in range(self.registry.num_voters)] + ["Total"])
            
            for i, item in enumerate(self.item_array):
                row = [
                    f"{self.tcr_array[i].get_tcr_value():.2f}",
                    str(item.is_valid()),
                    str(item.is_accepted())
                ]
                row += [f"{v.get_tokens():.2f}" for v in self.vote_results[i]]
                row.append(f"{sum(v.get_tokens() for v in self.vote_results[i]):.2f}")
                writer.writerow(row)

    def get_voter_status(self, voter):
        informed = "I" if voter.get_information() else "U"
        engaged = "E" if voter.get_engagement() else "U"
        return f"{informed}/{engaged}"

    def generate_plots(self):
        self.generate_voter_plot()
        self.generate_tcr_plot()

    def generate_voter_plot(self):
        categories = {
            'ie': {'values': [], 'count': 0},
            'iu': {'values': [], 'count': 0},
            'ue': {'values': [], 'count': 0},
            'uu': {'values': [], 'count': 0}
        }

        # Inicializa as contagens com base na primeira rodada
        if self.vote_results:
            for voter in self.vote_results[0]:
                key = ('i' if voter.get_information() else 'u') + ('e' if voter.get_engagement() else 'u')
                categories[key]['count'] += 1

        # Coleta dados históricos de cada rodada
        for round_voters in self.vote_results:
            sums = {k: 0.0 for k in categories}
            for voter in round_voters:  # Usa os eleitores da rodada específica
                #print(voter.voter_id)
                #print(voter.get_tokens())
                key = ('i' if voter.get_information() else 'u') + ('e' if voter.get_engagement() else 'u')
                sums[key] += voter.get_tokens()
            if (self.debug): 
                print("=" * 40)
                print(sums)
                    
            for key in categories:
                count = categories[key]['count'] or 1  # Evita divisão por zero
                #if (self.debug): 
                    #print(count)
                categories[key]['values'].append(sums[key] / count)

        # Plotagem do gráfico
        plt.figure(figsize=(12, 6))
        for label, data in categories.items():
            plt.plot(data['values'], label={
                'ie': 'Informed-Engaged',
                'iu': 'Informed-Unengaged',
                'ue': 'Uninformed-Engaged',
                'uu': 'Uninformed-Unengaged'
            }[label])
        
        plt.xlabel("Voting Round")
        plt.ylabel("Average Tokens")
        plt.legend()
        plt.savefig("voter_tokens.png")
        plt.close()

    def generate_tcr_plot(self):
        values = [t.get_tcr_value() for t in self.tcr_array]
        plt.plot(values)
        plt.xlabel("Voting Round")
        plt.ylabel("TCR Value")
        plt.savefig("tcr_values.png")
        plt.close()

    def create_voter_group(self, 
                          count: int,
                          tokens: float = None,
                          prob_engaged: float = None,  
                          prob_informed: float = None,
                          is_engaged: bool = None,
                          is_informed: bool = None
                          ): 
        """Create voter group with EXPLICIT probabilities"""
        
        # Validate inputs
        if count <= 0:
            raise ValueError("Count must be positive integer")
            
        if tokens is None:
            tokens = self.registry.default_tokens

        # Create voters with explicit probabilities
        for _ in range(count):
            voter = Voter(
                tokens=tokens,
                p_engaged=prob_engaged,
                p_informed=prob_informed,
                voter_id=self.next_voter_id,
                is_engaged=is_engaged,
                is_informed=is_informed
            )
            self.next_voter_id += 1
            self.voters.append(voter)

        self.registry.num_voters += count


if __name__ == "__main__":
    random.seed(42)
    prob = Probability(
        prob_vote_engaged=0.8,
        prob_vote_disengaged=0.2,
        prob_vote_correct_informed=0.7,
        prob_vote_correct_uninformed=0.5,
        prob_obj_engaged=0.7,
        prob_obj_disengaged=0.1,
        prob_obj_correct_informed=0.8,
        prob_obj_correct_uninformed=0.3,
        prob_engaged=0.5,
        prob_informed=0.5
    )

    registry = Registry(
        num_voters=40,
        default_tokens=1000,
        stake=200,
        initial_stake=200,
        delta=0.05,
        code_complexity=pow(2,2)
    )


    quantity_votes = registry.num_voters * (prob.prob_engaged * prob.prob_vote_engaged + (1 - prob.prob_engaged) * prob.prob_vote_disengaged)

    qp = registry.num_voters//4
    sim = Simulation(prob, registry, quantity_votes=registry.num_voters, debug=True)
 
    # IE - Informed Engaged
    sim.create_voter_group(
        count=2*qp,   
        tokens=3000,
        is_engaged=True,
        is_informed=True  
    )

    # UE - Uninformed Engaged
    sim.create_voter_group(
        count=qp,  
        tokens=3000,
        is_engaged=True,
        is_informed=False  
    )

    # UU - Uninformed Ungaged
    sim.create_voter_group(
        count=qp,   
        tokens=3000,
        is_engaged=False,
        is_informed=False 
    )

     # IU - Informed Ungaged
    sim.create_voter_group(
        count=qp,   
        tokens=3000,
        is_engaged=False,
        is_informed=True
    )

    qnt = 100

    for i in range(qnt):
        sim.run()
    print("accuracy: ", sim.valid_outcome/qnt * 100, "%")