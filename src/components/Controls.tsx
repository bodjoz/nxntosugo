import React from 'react';
import { Player } from '../game/TorusGo';

interface ControlsProps {
    boardSize: number;
    setBoardSize: (size: number) => void;
    onPass: () => void;
    onReset: () => void;
    currentPlayer: Player;
    scores: { black: number, white: number };
    captures: { '1': number, '-1': number };
    gameStatus: string;
    aiPlayer: string;
    setAiPlayer: (player: string) => void;
    showHeatmap: boolean;
    setShowHeatmap: (show: boolean) => void;
}

const Controls: React.FC<ControlsProps> = ({ boardSize, setBoardSize, onPass, onReset, currentPlayer, scores, captures, gameStatus, aiPlayer, setAiPlayer, showHeatmap, setShowHeatmap }) => {
    return (
        <div className="controls-panel">
            <div className="status-banner">
                <h2>{gameStatus}</h2>
                {gameStatus === 'Playing' && (
                    <p>Current Turn: <span className={currentPlayer === 1 ? 'black-turn' : 'white-turn'}>
                        {currentPlayer === 1 ? 'Black' : 'White'}
                    </span></p>
                )}
            </div>

            <div className="score-board">
                <div className="score-card black-score">
                    <h3>Black</h3>
                    <p>Territory: {scores.black}</p>
                    <p>Captures: {captures['1']}</p>
                </div>
                <div className="score-card white-score">
                    <h3>White</h3>
                    <p>Territory: {scores.white}</p>
                    <p>Captures: {captures['-1']}</p>
                </div>
            </div>

            <div className="settings">
                <label>
                    Board Size (n={boardSize}):
                    <input
                        type="range"
                        min="4"
                        max="19"
                        value={boardSize}
                        onChange={(e) => setBoardSize(parseInt(e.target.value, 10))}
                    />
                </label>
                <label>
                    AI Opponent:
                    <select value={aiPlayer} onChange={(e) => setAiPlayer(e.target.value)}>
                        <option value="none">None (Local PvP)</option>
                        <option value="1">Black</option>
                        <option value="-1">White</option>
                    </select>
                </label>
                <label className="heatmap-toggle">
                    <input
                        type="checkbox"
                        checked={showHeatmap}
                        onChange={(e) => setShowHeatmap(e.target.checked)}
                    />
                    Show AI Heatmap
                </label>
            </div>

            <div className="actions">
                <button onClick={onPass} className="btn pass-btn">Pass Turn</button>
                <button onClick={onReset} className="btn reset-btn">Reset Game</button>
            </div>
        </div>
    );
};

export default Controls;
