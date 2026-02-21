import torch
import numpy as np
from torusgo import TorusGo
from mcts import MCTS
from network import TorusGoNet

def play_game(model, mcts_simulations=40, temperature=1.0, device='cpu'):
    """Plays a single game of 4x4 Torus Go using MCTS and returns training data."""
    game = TorusGo(size=4)
    mcts = MCTS(model, num_simulations=mcts_simulations, device=device)
    
    states = []
    policies = []
    current_players = []
    
    while not game.game_over:
        # Determine temperature for this move (add randomness early on, then greedy)
        temp = temperature if game.passes_in_row == 0 and len(states) < 15 else 0.1
        
        # Get action probabilities from MCTS
        action_probs = mcts.get_action_prob(game, temperature=temp)
        
        # Store state and target policy
        states.append(game.get_state_input())
        policies.append(action_probs)
        current_players.append(game.current_player)
        
        # Choose action
        if temp == 0.1:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
            
        game.step(action)
        
    # Game over, compute rewards
    # get_reward returns 1 if Player 1 wins, -1 if Player -1 wins, 0 for draw
    final_reward = game.get_reward()
    
    # Generate value targets:
    # If a state was reached when it was player P's turn, the value target should be
    # +1 if player P won, -1 if player P lost.
    values = []
    for p in current_players:
        if p == 1:
            values.append(final_reward)
        else:
            values.append(-final_reward)
            
    return states, policies, values
