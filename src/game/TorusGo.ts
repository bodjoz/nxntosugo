export type Player = 1 | -1; // 1 for Black, -1 for White
export type Point = 0 | 1 | -1; // 0 for Empty, 1 for Black, -1 for White

export interface MoveResult {
    success: boolean;
    capturedStones: number[];
    error?: string;
}

export class TorusGo {
    public boardSize: number;
    public board: Point[];
    public currentPlayer: Player;
    public history: Set<string>; // For superko detection
    public moveCount: number;
    public passesInRow: number;
    public isGameOver: boolean;
    public captures: { '1': number; '-1': number };

    constructor(boardSize: number = 9) {
        this.boardSize = boardSize;
        this.board = new Array(boardSize * boardSize).fill(0);
        this.currentPlayer = 1;
        this.history = new Set();
        this.history.add(this.getBoardHash(this.board, this.currentPlayer));
        this.moveCount = 0;
        this.passesInRow = 0;
        this.isGameOver = false;
        this.captures = { '1': 0, '-1': 0 };
    }

    public getIndex(x: number, y: number): number {
        const wrappedX = ((x % this.boardSize) + this.boardSize) % this.boardSize;
        const wrappedY = ((y % this.boardSize) + this.boardSize) % this.boardSize;
        return wrappedY * this.boardSize + wrappedX;
    }

    public getCoords(index: number): [number, number] {
        return [index % this.boardSize, Math.floor(index / this.boardSize)];
    }

    // Gets the 4 cardinal neighbors wrapped on a torus
    public getNeighbors(index: number): number[] {
        const [x, y] = this.getCoords(index);
        return [
            this.getIndex(x, y - 1),
            this.getIndex(x - 1, y),
            this.getIndex(x + 1, y),
            this.getIndex(x, y + 1)
        ];
    }

    public getBoardHash(board: Point[], nextPlayer: Player): string {
        return board.join(',') + '|' + nextPlayer;
    }

    // Find all stones connected to the given index of the same color
    public findGroup(startIndex: number, boardState: Point[] = this.board): Set<number> {
        const color = boardState[startIndex];
        if (color === 0) return new Set();

        const group = new Set<number>();
        const stack = [startIndex];

        while (stack.length > 0) {
            const current = stack.pop()!;
            if (!group.has(current)) {
                group.add(current);
                const neighbors = this.getNeighbors(current);
                for (const neighbor of neighbors) {
                    if (boardState[neighbor] === color && !group.has(neighbor)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        return group;
    }

    // Count liberties for a given group of stones
    public getLiberties(group: Set<number>, boardState: Point[] = this.board): Set<number> {
        const liberties = new Set<number>();
        for (const index of group) {
            const neighbors = this.getNeighbors(index);
            for (const neighbor of neighbors) {
                if (boardState[neighbor] === 0) {
                    liberties.add(neighbor);
                }
            }
        }
        return liberties;
    }

    // Attempt to play a move
    public playMove(x: number, y: number): MoveResult {
        if (this.isGameOver) return { success: false, error: 'Game is over', capturedStones: [] };

        const index = this.getIndex(x, y);
        if (this.board[index] !== 0) {
            return { success: false, error: 'Point is not empty', capturedStones: [] };
        }

        // Clone board to test the move
        const newBoard = [...this.board];
        newBoard[index] = this.currentPlayer;

        const opponent: Player = this.currentPlayer === 1 ? -1 : 1;
        let capturedStonesSet = new Set<number>(); // Use a Set to collect captured stones

        // Check adjacent opponent stones for capture
        const neighbors = this.getNeighbors(index);
        for (const neighbor of neighbors) {
            if (newBoard[neighbor] === opponent) {
                const enemyGroup = this.findGroup(neighbor, newBoard);
                const liberties = this.getLiberties(enemyGroup, newBoard);
                if (liberties.size === 0) {
                    // Capture!
                    for (const stone of enemyGroup) {
                        newBoard[stone] = 0;
                        capturedStonesSet.add(stone); // Add to Set
                    }
                }
            }
        }

        // Convert the Set of captured stones to an array
        const capturedStones = Array.from(capturedStonesSet);

        // Ensure we don't capture ourselves unless it's a legal suicide (not typical in standard Go, but let's disallow suicide that doesn't capture)
        const myGroup = this.findGroup(index, newBoard);
        const myLiberties = this.getLiberties(myGroup, newBoard);

        if (myLiberties.size === 0) {
            return { success: false, error: 'Suicide is not allowed', capturedStones: [] };
        }

        // Superko check
        const boardHash = this.getBoardHash(newBoard, opponent);
        if (this.history.has(boardHash)) {
            return { success: false, error: 'Move violates Superko rule', capturedStones: [] };
        }

        // Apply move
        this.board = newBoard;
        this.currentPlayer = opponent;
        this.history.add(boardHash);
        this.moveCount++;
        this.passesInRow = 0;

        // Update captures
        this.captures[this.currentPlayer === 1 ? '-1' : '1'] += capturedStones.length;

        return { success: true, capturedStones };
    }

    public pass(): void {
        if (this.isGameOver) return;
        this.passesInRow++;
        if (this.passesInRow >= 2) {
            this.isGameOver = true;
        } else {
            this.currentPlayer = this.currentPlayer === 1 ? -1 : 1;
            this.history.add(this.getBoardHash(this.board, this.currentPlayer));
            this.moveCount++;
        }
    }

    public reset(): void {
        this.board = new Array(this.boardSize * this.boardSize).fill(0);
        this.currentPlayer = 1;
        this.history = new Set();
        this.history.add(this.getBoardHash(this.board, this.currentPlayer));
        this.moveCount = 0;
        this.passesInRow = 0;
        this.isGameOver = false;
        this.captures = { '1': 0, '-1': 0 };
    }

    // Tromp-Taylor Area Scoring
    public calculateScore(): { black: number; white: number } {
        const score = { black: 0, white: 0 };
        const visited = new Set<number>();

        for (let i = 0; i < this.board.length; i++) {
            if (this.board[i] === 1) score.black++;
            else if (this.board[i] === -1) score.white++;
            else if (!visited.has(i)) {
                // Empty intersection, find reachable stones
                const emptyGroup = new Set<number>();
                const stack = [i];
                let reachedBlack = false;
                let reachedWhite = false;

                while (stack.length > 0) {
                    const current = stack.pop()!;
                    if (!emptyGroup.has(current)) {
                        emptyGroup.add(current);
                        visited.add(current);
                        const neighbors = this.getNeighbors(current);
                        for (const n of neighbors) {
                            if (this.board[n] === 0 && !emptyGroup.has(n)) {
                                stack.push(n);
                            } else if (this.board[n] === 1) {
                                reachedBlack = true;
                            } else if (this.board[n] === -1) {
                                reachedWhite = true;
                            }
                        }
                    }
                }

                if (reachedBlack && !reachedWhite) score.black += emptyGroup.size;
                else if (reachedWhite && !reachedBlack) score.white += emptyGroup.size;
                // If it reached both, it's neutral (dame)
            }
        }

        return score;
    }
}
