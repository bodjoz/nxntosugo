import numpy as np

class TorusGo:
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
        
    def _get_hash(self, board, player):
        return board.tobytes() + bytes([player + 1])
        
    def get_legal_moves(self):
        """Returns a 1D array of legal moves. Length = size*size + 1 (pass is last)."""
        legal = np.zeros(self.size * self.size + 1, dtype=np.float32)
        if self.game_over:
            return legal
            
        # Pass is always legal unless game is over
        legal[-1] = 1.0
        
        # Check all intersections
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    idx = r * self.size + c
                    if self._is_legal(r, c):
                        legal[idx] = 1.0
        return legal
        
    def _get_neighbors(self, r, c):
        return [
            ((r - 1) % self.size, c),
            ((r + 1) % self.size, c),
            (r, (c - 1) % self.size),
            (r, (c + 1) % self.size)
        ]
        
    def _find_group(self, r, c, board=None):
        if board is None:
            board = self.board
        color = board[r, c]
        if color == 0:
            return set()
            
        group = set([(r, c)])
        stack = [(r, c)]
        
        while stack:
            curr_r, curr_c = stack.pop()
            for nr, nc in self._get_neighbors(curr_r, curr_c):
                if board[nr, nc] == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    stack.append((nr, nc))
        return group
        
    def _get_liberties(self, group, board=None):
        if board is None:
            board = self.board
        liberties = set()
        for r, c in group:
            for nr, nc in self._get_neighbors(r, c):
                if board[nr, nc] == 0:
                    liberties.add((nr, nc))
        return liberties
        
    def _is_legal(self, r, c):
        # 1. Must be empty
        if self.board[r, c] != 0:
            return False
            
        # Try move
        new_board = self.board.copy()
        new_board[r, c] = self.current_player
        opponent = -self.current_player
        
        captured_any = False
        
        # Check captures
        for nr, nc in self._get_neighbors(r, c):
            if new_board[nr, nc] == opponent:
                enemy_group = self._find_group(nr, nc, new_board)
                libs = self._get_liberties(enemy_group, new_board)
                if len(libs) == 0:
                    captured_any = True
                    for er, ec in enemy_group:
                        new_board[er, ec] = 0
                        
        # 2. Suicide rule
        if not captured_any:
            my_group = self._find_group(r, c, new_board)
            my_libs = self._get_liberties(my_group, new_board)
            if len(my_libs) == 0:
                return False
                
        # 3. Superko
        state_hash = self._get_hash(new_board, opponent)
        if state_hash in self.history:
            return False
            
        return True

    def step(self, action):
        """Action is 0 to size*size-1 for points, size*size for pass"""
        if self.game_over:
            return self.get_reward(), True
            
        if action == self.size * self.size:
            # Pass
            self.passes_in_row += 1
            if self.passes_in_row >= 2:
                self.game_over = True
            self.current_player = -self.current_player
            self.history.add(self._get_hash(self.board, self.current_player))
            return 0.0, self.game_over
            
        self.passes_in_row = 0
        r = action // self.size
        c = action % self.size
        
        self.board[r, c] = self.current_player
        opponent = -self.current_player
        
        # Resolve captures
        for nr, nc in self._get_neighbors(r, c):
            if self.board[nr, nc] == opponent:
                enemy_group = self._find_group(nr, nc)
                libs = self._get_liberties(enemy_group)
                if len(libs) == 0:
                    for er, ec in enemy_group:
                        self.board[er, ec] = 0
                        
        self.current_player = opponent
        self.history.add(self._get_hash(self.board, self.current_player))
        
        return 0.0, False # Game only ends on double pass
        
    def get_reward(self):
        """Tromp-Taylor Score from perspective of Player 1 (Black). Returns 1 (win), -1 (loss), 0 (draw)."""
        if not self.game_over:
            return 0.0
            
        black_score = 0
        white_score = 0
        visited = set()
        
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 1:
                    black_score += 1
                elif self.board[r, c] == -1:
                    white_score += 1
                elif (r, c) not in visited:
                    # Empty group
                    empty_group = set([(r, c)])
                    stack = [(r, c)]
                    visited.add((r, c))
                    
                    reached_black = False
                    reached_white = False
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        for nr, nc in self._get_neighbors(curr_r, curr_c):
                            if self.board[nr, nc] == 0 and (nr, nc) not in empty_group:
                                empty_group.add((nr, nc))
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                            elif self.board[nr, nc] == 1:
                                reached_black = True
                            elif self.board[nr, nc] == -1:
                                reached_white = True
                                
                    if reached_black and not reached_white:
                        black_score += len(empty_group)
                    elif reached_white and not reached_black:
                        white_score += len(empty_group)
                        
        if black_score > white_score: return 1.0
        elif white_score > black_score: return -1.0
        return 0.0

    def clone(self):
        new_game = TorusGo(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.history = set(self.history)
        new_game.passes_in_row = self.passes_in_row
        new_game.game_over = self.game_over
        return new_game
        
    def get_state_input(self):
        """Returns state tensor representation for Neural Network. 
        Shape: [2, size, size]
        Channel 0: Current player stones
        Channel 1: Opponent stones
        """
        state = np.zeros((2, self.size, self.size), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state
