import math
import numpy as np
import torch

class Node:
    def __init__(self, prior, parent=None, action=None, current_player=1):
        self.prior = prior
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.current_player = current_player
        
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
        
    def expand(self, game_state, policy):
        """Expand node. policy is 1D array [size*size + 1]"""
        # Optimized: only get legal move mask if we actually have potential children
        legal_actions = game_state.get_legal_moves()
        
        # Mask illegal actions
        legal_policy = policy * legal_actions
        sum_policy = np.sum(legal_policy)
        
        if sum_policy > 0:
            legal_policy /= sum_policy
        else:
            # If all policy moves illegal, fallback to uniform over legal
            legal_policy = legal_actions / np.sum(legal_actions)
            
        for action in np.where(legal_policy > 0)[0]:
            self.children[int(action)] = Node(
                prior=float(legal_policy[action]),
                parent=self,
                action=int(action),
                current_player=-self.current_player
            )

class MCTS:
    def __init__(self, model, num_simulations=40, c_puct=1.5, device='cpu',
                 add_dirichlet_noise=False, dirichlet_alpha=0.03, dirichlet_epsilon=0.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.in_channels = getattr(model, 'in_channels', 2)
        
    def _ucb_score(self, parent_visit_sqrt, child):
        # Optimized: pass parent_visit_sqrt to avoid redundant sqrt calls
        prior_score = self.c_puct * child.prior * parent_visit_sqrt / (child.visit_count + 1)
        expected_value = -child.value
        return expected_value + prior_score

    def _get_state_tensor(self, game, move_number=0):
        state = game.get_state_input(in_channels=self.in_channels, move_number=move_number)
        # Fast transfer to device
        return torch.from_numpy(state).unsqueeze(0).to(self.device, non_blocking=True)
        
    def get_action_prob(self, game, temperature=1.0, move_number=0):
        root = Node(0, current_player=game.current_player)
        
        # Initial expansion
        state_tensor = self._get_state_tensor(game, game.moves_made)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(state_tensor)
            policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        root.expand(game, policy)
        
        if self.add_dirichlet_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for i, action in enumerate(actions):
                root.children[action].prior = (1 - self.dirichlet_epsilon) * root.children[action].prior + self.dirichlet_epsilon * noise[i]
        
        # Simulations
        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone() # Clones are now fast (no history)
            
            # 1. Select
            while node.children:
                parent_visit_sqrt = math.sqrt(node.visit_count)
                # Fast child selection
                action, next_node = max(
                    node.children.items(), 
                    key=lambda item: self._ucb_score(parent_visit_sqrt, item[1])
                )
                scratch_game.step(action)
                node = next_node
                
            # 2. Evaluate & Expand
            if not scratch_game.game_over:
                state_tensor = self._get_state_tensor(scratch_game, scratch_game.moves_made)
                with torch.no_grad():
                    policy, value_tensor = self.model(state_tensor)
                    policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
                    value = value_tensor.item()
                node.expand(scratch_game, policy)
            else:
                reward = scratch_game.get_reward()
                # value is from point of view of current_player
                value = reward if scratch_game.current_player == 1 else -reward
                    
            # 3. Backpropagate
            v = value
            curr = node
            while curr is not None:
                curr.value_sum += v
                curr.visit_count += 1
                v = -v
                curr = curr.parent
                
        # Generate Action Probabilities
        action_probs = np.zeros(game.total_points + 1, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
            
        sum_visits = np.sum(action_probs)
        if sum_visits > 0:
            if temperature == 0:
                best_action = np.argmax(action_probs)
                action_probs = np.zeros_like(action_probs)
                action_probs[best_action] = 1.0
            else:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs /= np.sum(action_probs)
        else:
            action_probs = game.get_legal_moves()
            action_probs /= np.sum(action_probs)
            
        return action_probs
