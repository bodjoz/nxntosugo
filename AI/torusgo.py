import numpy as np

class TorusGo:
    # 1D index optimized neighbors
    _neighbor_cache = {}

    def __init__(self, size=9, max_moves=None):
        self.size = size
        self.total_points = size * size
        # Board as 1D array: 0=empty, 1=Black, -1=White
        self.board = np.zeros(self.total_points, dtype=np.int8)
        self.current_player = 1
        self.history = set()
        self.board_hash = self.board.tobytes()
        self.history.add(self.board_hash + bytes([self.current_player + 1]))
        self.passes_in_row = 0
        self.game_over = False
        self.moves_made = 0
        self.max_moves = max_moves or (size * size * 2)
        
        # Simple Ko: store the previous board hash to prevent immediate recapture
        self.prev_board_hash = None
        
        if size not in TorusGo._neighbor_cache:
            cache = []
            for i in range(self.total_points):
                r, c = i // size, i % size
                neighbors = [
                    ((r - 1) % size) * size + c,
                    ((r + 1) % size) * size + c,
                    r * size + ((c - 1) % size),
                    r * size + ((c + 1) % size)
                ]
                cache.append(np.array(neighbors, dtype=np.int32))
            TorusGo._neighbor_cache[size] = cache
        self.neighbor_indices = TorusGo._neighbor_cache[size]

    def get_legal_moves(self):
        """Ultralight legality check for MCTS. Skips Superko, only checks suicide/Ko."""
        legal = np.zeros(self.total_points + 1, dtype=np.float32)
        if self.game_over: return legal
        legal[-1] = 1.0 # Pass
        
        for i in range(self.total_points):
            if self.board[i] == 0:
                if self._is_legal_quick(i):
                    legal[i] = 1.0
        return legal

    def _is_legal_quick(self, pos):
        """Checks legality without board copies or hashes where possible."""
        color = self.current_player
        opponent = -color
        
        # 1. Immediate liberty? (Fastest check)
        for neighbor in self.neighbor_indices[pos]:
            if self.board[neighbor] == 0:
                return True
        
        # 2. Capture an enemy?
        for neighbor in self.neighbor_indices[pos]:
            if self.board[neighbor] == opponent:
                _, libs = self._find_group_info_fast(neighbor, self.board, ignore_pos=pos)
                if libs == 0:
                    # Potential capture. For MCTS, we consider this legal.
                    # Simple Ko check would go here if needed.
                    return True
                    
        # 3. Suicide?
        # If no immediate liberties and no captures, check if own group will have liberties
        # Optimization: temporarily place stone and check group info
        self.board[pos] = color
        _, libs = self._find_group_info_fast(pos, self.board)
        self.board[pos] = 0 # Reset
        
        return libs > 0

    def _find_group_info_fast(self, start_pos, board, ignore_pos=-1):
        """Returns group and count of UNIQUE liberties."""
        color = board[start_pos]
        group = [start_pos]
        visited = {start_pos}
        stack = [start_pos]
        liberties = set()
        
        while stack:
            curr = stack.pop()
            for neighbor in self.neighbor_indices[curr]:
                if neighbor == ignore_pos: continue
                val = board[neighbor]
                if val == 0:
                    liberties.add(neighbor)
                elif val == color and neighbor not in visited:
                    visited.add(neighbor)
                    group.append(neighbor)
                    stack.append(neighbor)
        return group, len(liberties)

    def step(self, action):
        if self.game_over: return 0.0, True
        self.moves_made += 1
        
        if action == self.total_points or self.moves_made >= self.max_moves:
            # Pass or Max Moves
            self.passes_in_row += 1
            if self.passes_in_row >= 2 or self.moves_made >= self.max_moves:
                self.game_over = True
            self.current_player = -self.current_player
            self.board_hash = self.board.tobytes()
            self.history.add(self.board_hash + bytes([self.current_player + 1]))
            return 0.0, self.game_over
            
        self.passes_in_row = 0
        color = self.current_player
        opponent = -color
        self.board[action] = color
        
        # Resolve captures
        for neighbor in self.neighbor_indices[action]:
            if self.board[neighbor] == opponent:
                group, libs = self._find_group_info_fast(neighbor, self.board)
                if libs == 0:
                    for p in group: self.board[p] = 0
                        
        self.current_player = opponent
        self.board_hash = self.board.tobytes()
        self.history.add(self.board_hash + bytes([self.current_player + 1]))
        return 0.0, False

    def get_reward(self):
        if not self.game_over: return 0.0
        black_score = 0
        white_score = 6.5
        visited = [False] * self.total_points
        for i in range(self.total_points):
            if self.board[i] == 1: black_score += 1
            elif self.board[i] == -1: white_score += 1
            elif not visited[i]:
                group, reached_black, reached_white = self._score_empty_group(i, visited)
                if reached_black and not reached_white: black_score += len(group)
                elif reached_white and not reached_black: white_score += len(group)
        return 1.0 if black_score > white_score else -1.0 if white_score > black_score else 0.0

    def _score_empty_group(self, start_pos, globally_visited):
        group = [start_pos]
        stack = [start_pos]
        globally_visited[start_pos] = True
        reached_black, reached_white = False, False
        while stack:
            curr = stack.pop()
            for neighbor in self.neighbor_indices[curr]:
                val = self.board[neighbor]
                if val == 0:
                    if not globally_visited[neighbor]:
                        globally_visited[neighbor] = True
                        group.append(neighbor)
                        stack.append(neighbor)
                elif val == 1: reached_black = True
                elif val == -1: reached_white = True
        return group, reached_black, reached_white

    def clone(self):
        new_game = TorusGo(self.size, self.max_moves)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.history = self.history.copy()
        new_game.passes_in_row = self.passes_in_row
        new_game.game_over = self.game_over
        new_game.moves_made = self.moves_made
        return new_game

    def get_state_input(self, in_channels=2, move_number=0):
        board_2d = self.board.reshape(self.size, self.size)
        if in_channels == 4:
            state = np.zeros((4, self.size, self.size), dtype=np.float32)
            state[0] = (board_2d == self.current_player).astype(np.float32)
            state[1] = (board_2d == -self.current_player).astype(np.float32)
            state[2] = 1.0 if self.current_player == 1 else 0.0
            state[3] = min(1.0, self.moves_made / (2 * self.total_points))
        else:
            state = np.zeros((2, self.size, self.size), dtype=np.float32)
            state[0] = (board_2d == self.current_player).astype(np.float32)
            state[1] = (board_2d == -self.current_player).astype(np.float32)
        return state
