import pandas as pd
import pulp
import numpy as np
import matplotlib.pyplot as plt

def Maxf1f2(w, df1, df2, df3):
    df1['Index'] = df1.index
    df2['Index'] = df2.index
    df3['Index'] = df3.index

    # Define fully evolved Pokémon
    pokemon_fully_evolved = [p for p in df1.index if df1.loc[p, "Fully Evolved"] == 1]
    df1 = df1.loc[pokemon_fully_evolved]

    # Define allowed moves
    allowed_moved_per_pokemon = {p: [m for m in df2.index if df3.loc[p, str(m)] == 1] for p in pokemon_fully_evolved}
    allowed_moved_per_pokemon_tuples = [(p, m) for p in pokemon_fully_evolved for m in allowed_moved_per_pokemon[p]]

    # Define attacker types
    df1['Attacker_Type'] = df1.apply(
        lambda row: 'Physical' if row['Attack'] > row['SpAttack'] else (
            'Special' if row['SpAttack'] > row['Attack'] else 'Both'),
        axis=1
    )

    # Define Pokémon types and resistances
    types = ['Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic',
             'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']
    for attack_type in types:
        df1[f'Resists_{attack_type}'] = df1[attack_type] < 1
    resists_cols = [f'Resists_{attack_type}' for attack_type in types]

    # Set index for moves and learnset
    moves = df2.set_index('Index')
    learnset = df3.set_index('Index')

    # Decision variables
    X = pulp.LpVariable.dicts("X", pokemon_fully_evolved, 0, 1, pulp.LpBinary)
    Y = pulp.LpVariable.dicts("Y", allowed_moved_per_pokemon_tuples, 0, 1, pulp.LpBinary)

    # Weights
    wHP = 1
    wAttack = 1
    wSpAttack = 1
    wDefense = 1
    wSpDefense = 1
    wSpeed = 1

    # Objective function f1
    f1 = pulp.lpSum([
        X[p] * (
                wHP * df1.loc[p, 'HP'] +
                wAttack * df1.loc[p, 'Attack'] +
                wSpAttack * df1.loc[p, 'SpAttack'] +
                wDefense * df1.loc[p, 'Defense'] +
                wSpDefense * df1.loc[p, 'SpDefense'] +
                wSpeed * df1.loc[p, 'Speed']
        ) for p in pokemon_fully_evolved
    ])

    # Objective function f2
    moves['Expected_Power'] = moves['Power'] * (moves['Accuracy'] / 100)
    f2 = pulp.lpSum([
        Y[(p, m)] * moves.loc[m, 'Expected_Power'] * (
                1 + 0.5 * (moves.loc[m, 'Type'] in [df1.loc[p, 'Type1'], df1.loc[p, 'Type2']])
        )
        for p, m in allowed_moved_per_pokemon_tuples
    ])

    # Combine objectives
    prob = pulp.LpProblem("MaximizeObjective", pulp.LpMaximize)
    objective_function = w * f1 + (1 - w) * f2
    prob += objective_function

    # Add constraints
    prob += pulp.lpSum([X[p] for p in pokemon_fully_evolved]) == 6  # Select exactly 6 Pokémon

    for num in df1['Number'].unique():
        prob += pulp.lpSum([X[p] for p in pokemon_fully_evolved if df1.loc[p, 'Number'] == num]) <= 1

    prob += pulp.lpSum(
        [X[p] for p in df1[(df1['Attacker_Type'] == 'Physical') | (df1['Attacker_Type'] == 'Both')]['Index']]) >= 1
    prob += pulp.lpSum(
        [X[p] for p in df1[(df1['Attacker_Type'] == 'Special') | (df1['Attacker_Type'] == 'Both')]['Index']]) >= 1
    prob += pulp.lpSum([X[p] for p in df1[(df1['Legendary'] == 1) | (df1['Mythical'] == 1)][
        'Index']]) <= 2, "Max 2 Legendary/Mythical Pokémon"

    for attack_type in types:
        prob += pulp.lpSum([X[p] * df1.loc[p, f'Resists_{attack_type}'] for p in
                            pokemon_fully_evolved]) >= 1, f"At least one resists {attack_type}"

    for p in pokemon_fully_evolved:
        prob += pulp.lpSum(Y[(p, m)] for m in allowed_moved_per_pokemon[p]) == 4 * X[p]

    for m in df2.index:
        prob += pulp.lpSum(Y[(p, m)] for p in pokemon_fully_evolved if m in allowed_moved_per_pokemon[p]) <= 1

    # Solve the problem
    prob.solve()

    # Get results
    selected_pokemon = [df1.loc[p, 'Name'] for p in pokemon_fully_evolved if X[p].value() == 1]
    selected_moves = {
        df1.loc[p, 'Name']: [moves.loc[m, 'Name'] for m in allowed_moved_per_pokemon[p] if Y[(p, m)].value() == 1] for p
        in pokemon_fully_evolved if X[p].value() == 1}

    return selected_pokemon, selected_moves, f1.value(), f2.value()


# Load data
df1 = pd.read_csv('F:/pycharm/biyesheji/pokemon_swsh_data.csv')
df2 = pd.read_csv('F:/pycharm/biyesheji/attack_swsh_data.csv')
df3 = pd.read_csv('F:/pycharm/biyesheji/learnset.csv')

# Calculate tilde_f1 and tilde_f2 for normalization
_, _, tilde_f1, _ = Maxf1f2(1, df1, df2, df3)  # Maximizing f1 (w=1)
_, _, _, tilde_f2 = Maxf1f2(0, df1, df2, df3)  # Maximizing f2 (w=0)

# Generate Pareto front
w_values = np.linspace(0, 1, 31)
f1_values = []
f2_values = []

for w in w_values:
    _, _, f1_val, f2_val = Maxf1f2(w, df1, df2, df3)
    f1_values.append(f1_val / tilde_f1)
    f2_values.append(f2_val / tilde_f2)

# Plot Pareto front
plt.plot(f1_values, f2_values, marker='o')
plt.xlabel('f1 / tilde(f1)')
plt.ylabel('f2 / tilde(f2)')
plt.title('Pareto Front')
plt.grid(True)
plt.show()

# Run the function with a specific weight
selected_pokemon, selected_moves, f1, f2 = Maxf1f2(0, df1, df2, df3)

# Output the results
print("Selected Pokémon:")
for p in selected_pokemon:
    print(p)

print("\nSelected Moves:")
for p, moves in selected_moves.items():
    print(f"{p}: {', '.join(moves)}")