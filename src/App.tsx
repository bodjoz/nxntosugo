import React, { useState, useEffect, useRef } from 'react';
import Board from './components/Board';
import Controls from './components/Controls';
import { TorusGo } from './game/TorusGo';
import './App.css';

function App() {
    const [boardSize, setBoardSize] = useState<number>(9);
    const gameRef = useRef<TorusGo>(new TorusGo(9));
    const [, setTick] = useState<number>(0);
    const [lastMove, setLastMove] = useState<[number, number] | null>(null);
    const [aiHeatmap, setAiHeatmap] = useState<number[] | undefined>(undefined);
    const [errorMessage, setErrorMessage] = useState<string>('');

    const forceUpdate = () => setTick(t => t + 1);

    useEffect(() => {
        gameRef.current = new TorusGo(boardSize);
        setLastMove(null);
        setAiHeatmap(undefined);
        setErrorMessage('');
        forceUpdate();
    }, [boardSize]);

    const handlePlayMove = (x: number, y: number) => {
        const game = gameRef.current;
        if (game.isGameOver) return;

        const result = game.playMove(x, y);
        if (!result.success) {
            setErrorMessage(result.error || 'Invalid move');
            setTimeout(() => setErrorMessage(''), 2000);
            return;
        }

        setLastMove([x, y]);
        setErrorMessage('');
        forceUpdate();

        // Trigger AI mock logic here if needed, or clear past heatmap
        if (aiHeatmap) evaluateAiHook();
    };

    const handlePass = () => {
        gameRef.current.pass();
        setLastMove(null);
        setErrorMessage('Passed');
        setTimeout(() => setErrorMessage(''), 2000);
        forceUpdate();
        if (aiHeatmap) evaluateAiHook();
    };

    const handleReset = () => {
        gameRef.current.reset();
        setLastMove(null);
        setAiHeatmap(undefined);
        forceUpdate();
    };

    const game = gameRef.current;
    const scores = game.calculateScore();
    let status = 'Playing';
    if (game.isGameOver) {
        status = scores.black > scores.white ? 'Black Wins!' : (scores.white > scores.black ? 'White Wins!' : 'Tie Game');
    }

    // AI evaluation placeholder function
    const evaluateAiHook = async () => {
        // In a real app we would pass board state to a worker
        // For now, doing nothing. User will manually trigger via file upload.
    };

    const handleAiUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                try {
                    const json = JSON.parse(evt.target?.result as string);
                    // Expecting { heatmap: [ ...n*n sized array... ] }
                    if (json.heatmap && Array.isArray(json.heatmap) && json.heatmap.length === boardSize * boardSize) {
                        setAiHeatmap(json.heatmap);
                        setErrorMessage('AI heatmap loaded successfully.');
                        setTimeout(() => setErrorMessage(''), 2000);
                    } else {
                        setErrorMessage('Invalid AI network format or size mismatch.');
                    }
                } catch (err) {
                    setErrorMessage('Error parsing AI JSON.');
                }
            };
            reader.readAsText(file);
        }
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>Toroidal Go Web UI</h1>
            </header>

            {errorMessage && <div className="error-toast">{errorMessage}</div>}

            <div className="main-content">
                <div className="board-section">
                    <Board
                        size={game.boardSize}
                        boardState={game.board}
                        currentPlayer={game.currentPlayer}
                        onPlayMove={handlePlayMove}
                        lastMove={lastMove}
                        aiHeatmap={aiHeatmap}
                    />
                </div>

                <div className="side-section">
                    <Controls
                        boardSize={boardSize}
                        setBoardSize={setBoardSize}
                        onPass={handlePass}
                        onReset={handleReset}
                        currentPlayer={game.currentPlayer}
                        scores={scores}
                        captures={game.captures}
                        gameStatus={status}
                    />

                    <div className="ai-controls panel">
                        <h3>AI Hook</h3>
                        <p>Upload a dummy network JSON to test heatmap evaluation visualization.</p>
                        <input type="file" accept=".json" onChange={handleAiUpload} className="file-input" />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
