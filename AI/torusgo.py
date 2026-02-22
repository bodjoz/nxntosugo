import numpy as np

class TorusGo:
    # Class-level cache for neighbors to avoid recalculating modulo every time
    _neighbor_cache = {}

    def __init__(self, size=4):
        self.size = size
        # Board representation: 
        # 0 = empty, 1 = Black, -1 = White
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1
        self.history = set()
        self.history.add(self._get_hash(self.board, self.current_player))
        self.passes_in_row = 0
        self.game_over = False
        
        # Initialize neighbor cache for this size if not present
        if size not in TorusGo._neighbor_cache:
            cache = []
            for r in range(size):
                row_neighbors = []
                for c in range(size):
                    row_neighbors.append([
                        ((r - 1) % size, c),
                        ((r + 1) % size, c),
                        (r, (c - 1) % size),
                        (r, (c + 1) % size)
                    ])
                cache.append(row_neighbors)
            TorusGo._neighbor_cache[size] = cache
        self.neighbors = TorusGo._neighbor_cache[size]
        
    def _get_hash(self, board, player):
        return board.tobytes() + bytes([player + 1])
        
    def get_legal_moves(self):
        """Returns a 1D array of legal moves. Length = size*size + 1 (pass is last)."""
        legal = np.zeros(self.size * self.size + 1, dtype=np.float32)
        if self.game_over:
            return legal
            
        legal[-1] = 1.0 # Pass
        
        # Check all intersections.
        for r in range(self.size):
            board_row = self.board[r]
            for c in range(self.size):
                if board_row[c] == 0:
                    if self._is_legal(r, c):
                        legal[r * self.size + c] = 1.0
        return legal
        
    def _find_group_info(self, r, c, board):
        """Finds group and its liberties in one pass."""
        color = board[r, c]
        group = {(r, c)}
        stack = [(r, c)]
        liberties = set()
        
        while stack:
            curr_r, curr_c = stack.pop()
            for nr, nc in self.neighbors[curr_r][curr_c]:
                val = board[nr, nc]
                if val == 0:
                    liberties.add((nr, nc))
                elif val == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    stack.append((nr, nc))
        return group, len(liberties)
        
    def _is_legal(self, r, c):
        if self.board[r, c] != 0:
            return False
            
        color = self.current_player
        opponent = -color
        
        has_direct_liberty = False
        enemy_neighbors = []
        
        # Quick check for direct liberties or potential captures
        for nr, nc in self.neighbors[r][c]:
            val = self.board[nr, nc]
            if val == 0:
                has_direct_liberty = True
            elif val == opponent:
                enemy_neighbors.append((nr, nc))
                
        # Try move on a temp board for capture resolution and suicide check
        captured_any = False
        new_board = None
        
        # Check neighbors to see if we capture them
        for nr, nc in enemy_neighbors:
            # We must use the current state of liberties
            # If we already captured a group containing this neighbor, skip
            if new_board is not None and new_board[nr, nc] == 0:
                continue
                
            group, libs = self._find_group_info(nr, nc, self.board)
            if libs == 1: # Filling the last liberty!
                captured_any = True
                if new_board is None:
                    new_board = self.board.copy()
                    new_board[r, c] = color
                for er, ec in group:
                    new_board[er, ec] = 0
                    
        if not captured_any:
            if not has_direct_liberty:
                # Suicide check: does our own group have any liberties?
                new_board_tmp = self.board.copy()
                new_board_tmp[r, c] = color
                _, libs = self._find_group_info(r, c, new_board_tmp)
                if libs == 0:
                    return False
                new_board = new_board_tmp
            else:
                # Definitely not suicide, just create the board for superko check
                new_board = self.board.copy()
                new_board[r, c] = color
        
        # Superko check
        state_hash = self._get_hash(new_board, opponent)
        return state_hash not in self.history

    def step(self, action):
        if self.game_over:
            return self.get_reward(), True
            
        if action == self.size * self.size:
            self.passes_in_row += 1
            if self.passes_in_row >= 2:
                self.game_over = True
            self.current_player = -self.current_player
            self.history.add(self._get_hash(self.board, self.current_player))
            return 0.0, self.game_over
            
        self.passes_in_row = 0
        r, c = action // self.size, action % self.size
        
        color = self.current_player
        opponent = -color
        self.board[r, c] = color
        
        # Resolve captures efficiently
        for nr, nc in self.neighbors[r][c]:
            if self.board[nr, nc] == opponent:
                group, libs = self._find_group_info(nr, nc, self.board)
                if libs == 0:
                    for er, ec in group:
                        self.board[er, ec] = 0
                        
        self.current_player = opponent
        self.history.add(self._get_hash(self.board, self.current_player))
        return 0.0, False
        
    def get_reward(self):
        if not self.game_over:
            return 0.0
            
        black_score = 0
        white_score = 6.5
        visited = set()
        
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 1:
                    black_score += 1
                elif self.board[r, c] == -1:
                    white_score += 1
                elif (r, c) not in visited:
                    group, reached_black, reached_white = self._score_empty_group(r, c, visited)
                    if reached_black and not reached_white:
                        black_score += len(group)
                    elif reached_white and not reached_black:
                        white_score += len(group)
                        
        return 1.0 if black_score > white_score else -1.0 if white_score > black_score else 0.0

    def _score_empty_group(self, r, c, globally_visited):
        group = {(r, c)}
        stack = [(r, c)]
        globally_visited.add((r, c))
        reached_black = False
        reached_white = False
        
        while stack:
            curr_r, curr_c = stack.pop()
            for nr, nc in self.neighbors[curr_r][curr_c]:
                val = self.board[nr, nc]
                if val == 0:
                    if (nr, nc) not in group:
                        group.add((nr, nc))
                        globally_visited.add((nr, nc))
                        stack.append((nr, nc))
                elif val == 1:
                    reached_black = True
                elif val == -1:
                    reached_white = True
        return group, reached_black, reached_white

    def clone(self):
        new_game = TorusGo(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.history = set(self.history)
        new_game.passes_in_row = self.passes_in_row
        new_game.game_over = self.game_over
        return new_game
        
    def get_state_input(self, in_channels=2, move_number=0):
        if in_channels == 4:
            state = np.zeros((4, self.size, self.size), dtype=np.float32)
            state[0] = (self.board == self.current_player).astype(np.float32)
            state[1] = (self.board == -self.current_player).astype(np.float32)
            state[2] = 1.0 if self.current_player == 1 else 0.0
            state[3] = min(1.0, move_number / (2 * self.size * self.size))
        else:
            state = np.zeros((2, self.size, self.size), dtype=np.float32)
            state[0] = (self.board == self.current_player).astype(np.float32)
            state[1] = (self.board == -self.current_player).astype(np.float32)
        return state
