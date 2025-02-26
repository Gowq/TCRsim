# TCRsim

TCRsim is a Token Curated Registry (TCR) simulation developed to evaluate a token-inflation mechanism for enhancing engagement in TCRs. The simulation allows modification of TCR parameters (number of voters, tokens, inflation rate) to analyze their impact on the TCR's overall value. Setting the inflation rate to zero simulates a traditional TCR.

The simulation generates items and voters and simulates a voting process based on voting likelihood and correct voting probability. Voters are categorized into four classes based on their engagement (engaged/unengaged) and information (informed/uninformed) status.

## Output

TCRsim generates:
1. Detailed logs of item outcomes and voter token holdings
2. Three visualization plots:
   - Average wealth distribution among voter classes
   - Total tokens per voter class
   - TCR value across multiple voting periods

## Prerequisites

- Python 3.x
- Required Python packages:
  - matplotlib
  - random
  - csv
  - copy

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Navigate to the project directory:
```bash
cd TCRsim
```

## Usage

You can run the simulation using:

```bash
python Simulation.py
```

### Configuration

The simulation can be configured by modifying the parameters in `Simulation.py`:

```python
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
```

### Voter Groups

The simulation supports creating different voter groups with specific characteristics:
- Informed-Engaged (IE)
- Uninformed-Engaged (UE)
- Uninformed-Unengaged (UU)
- Informed-Unengaged (IU)

Example of creating voter groups:

```python
sim.create_voter_group(
    count=10,   # Number of voters in group
    tokens=3000,
    is_engaged=True,
    is_informed=True  
)
```

## Components

- `Item.py`: Defines the item class for registry entries
- `Probability.py`: Handles probability calculations for voting and objections
- `Registry.py`: Manages the TCR parameters and state
- `Simulation.py`: Main simulation logic
- `TCR.py`: Implements TCR value calculations
- `Voter.py`: Defines voter behavior and characteristics

## Reference

This simulator was used to produce results for:
Yi Lucy Wang, Bhaskar Krishnamachari, "Enhancing Engagement in Token-Curated Registries via an Inflationary Mechanism," preprint manuscript, November 2018.
