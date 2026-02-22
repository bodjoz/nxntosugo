import torch
import numpy as np
from torusgo import TorusGo
from mcts import MCTS
from network import TorusGoNet

def play_game(model, mcts_simulations=40, temperature=1.0, device='cpu', size=4,
              add_noise=False, in_channels=2):
    """Plays a single game of Torus Go using MCTS and returns training data."""
    game = TorusGo(size=size)
    mcts = MCTS(model, num_simulations=mcts_simulations, device=device,
                add_dirichlet_noise=add_noise, dirichlet_alpha=0.03, dirichlet_epsilon=0.25)
    
    states = []
    policies = []
    current_players = []
    move_number = 0
    
    while not game.game_over:
        # Temperature schedule: explore for first 20 moves, then play nearly greedy
        temp = temperature if move_number < 20 else 0.1
        
        # Get action probabilities from MCTS
        action_probs = mcts.get_action_prob(game, temperature=temp, move_number=move_number)
        
        # Store state and target policy
        states.append(game.get_state_input(in_channels=in_channels, move_number=move_number))
        policies.append(action_probs)
        current_players.append(game.current_player)
        
        # Choose action
        if temp <= 0.1:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
            
        game.step(action)
        move_number += 1
        
    # Game over, compute rewards
    final_reward = game.get_reward()
    
    # Generate value targets
    values = []
    for p in current_players:
        if p == 1:
            values.append(final_reward)
        else:
            values.append(-final_reward)
            
    return states, policies, values


def play_game_for_worker(args):
    """Wrapper for multiprocessing Pool — loads model weights on CPU and plays one game.
    Returns (states, policies, values, game_result) where game_result is:
      +1.0 = Black wins, -1.0 = White wins, 0.0 = draw
    """
    model_state_dict, mcts_simulations, size, in_channels = args
    
    device = torch.device('cpu')
    model = TorusGoNet(size=size, channels=128, num_res_blocks=5, in_channels=in_channels).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    states, policies, values = play_game(
        model, mcts_simulations=mcts_simulations, temperature=1.0,
        device=device, size=size, add_noise=True, in_channels=in_channels
    )
    
    # Determine game result from value targets:
    # values[0] is from Black's perspective (Black always moves first)
    game_result = values[0] if len(values) > 0 else 0.0
    
    return states, policies, values, game_result


def augment_torus_data(states, policies, values, size=9, num_augments=4):
    """Apply toroidal shift augmentations to training data.
    
    On a torus, shifting all stones by (dr, dc) produces a valid, equivalent position.
    This gives us size*size = 81 symmetries for 9x9. We sample a subset for efficiency.
    """
    aug_states = list(states)
    aug_policies = list(policies)
    aug_values = list(values)
    
    for i in range(len(states)):
        state = states[i]  # shape: [C, size, size]
        policy = policies[i]  # shape: [size*size + 1]
        value = values[i]
        
        for _ in range(num_augments):
            dr = np.random.randint(0, size)
            dc = np.random.randint(0, size)
            if dr == 0 and dc == 0:
                continue
                
            # Shift the board state (all channels)
            new_state = np.roll(np.roll(state, dr, axis=1), dc, axis=2)
            
            # Shift the policy (board moves only, not the pass move)
            board_policy = policy[:size * size].reshape(size, size)
            new_board_policy = np.roll(np.roll(board_policy, dr, axis=0), dc, axis=1)
            new_policy = np.append(new_board_policy.flatten(), policy[-1])  # keep pass prob
            
            aug_states.append(new_state)
            aug_policies.append(new_policy)
            aug_values.append(value)
    
    return aug_states, aug_policies, aug_values
