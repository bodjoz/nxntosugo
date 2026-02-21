import React, { useState } from 'react';
import { Player, Point } from '../game/TorusGo';

interface BoardProps {
    size: number;
    boardState: Point[];
    currentPlayer: Player;
    onPlayMove: (x: number, y: number) => void;
    lastMove: [number, number] | null;
    aiHeatmap?: number[]; // Array of size n*n with values 0-1 for heatmap opacity
}

const Board: React.FC<BoardProps> = ({ size, boardState, currentPlayer, onPlayMove, lastMove, aiHeatmap }) => {
    const [hoverX, setHoverX] = useState<number | null>(null);
    const [hoverY, setHoverY] = useState<number | null>(null);

    const containerSize = 600;
    const padding = 20;
    const gridVisualSize = containerSize - 2 * padding;
    // 2n x 2n grid => 2n - 1 cells across
    // However, Go board intersections are the lines themselves.
    // There are 2n intersections, spanning from 0 to 2n - 1.
    const cellSize = gridVisualSize / (2 * size - 1);

    const doubleSize = 2 * size;

    // Center highlighting box
    const highlightStart = Math.floor(size / 2);
    const highlightSize = size;
    const highlightX = padding + highlightStart * cellSize - cellSize / 2;
    const highlightY = padding + highlightStart * cellSize - cellSize / 2;
    const highlightSquareWidth = highlightSize * cellSize;

    const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left - padding;
        const y = e.clientY - rect.top - padding;

        let nearestX = Math.round(x / cellSize);
        let nearestY = Math.round(y / cellSize);

        if (nearestX >= 0 && nearestX < doubleSize && nearestY >= 0 && nearestY < doubleSize) {
            setHoverX(nearestX % size);
            setHoverY(nearestY % size);
        } else {
            setHoverX(null);
            setHoverY(null);
        }
    };

    const handlePointerLeave = () => {
        setHoverX(null);
        setHoverY(null);
    };

    const handleClick = () => {
        if (hoverX !== null && hoverY !== null) {
            onPlayMove(hoverX, hoverY);
        }
    };

    const stones = [];
    const lines = [];
    const heatmaps = [];

    // Draw Grid Lines
    for (let i = 0; i < doubleSize; i++) {
        const pos = padding + i * cellSize;
        lines.push(
            <line key={`v-${i}`} x1={pos} y1={padding} x2={pos} y2={containerSize - padding} stroke="#666" strokeWidth={1} />,
            <line key={`h-${i}`} x1={padding} y1={pos} x2={containerSize - padding} y2={pos} stroke="#666" strokeWidth={1} />
        );
    }

    // Draw Stones & Heatmaps
    for (let vx = 0; vx < doubleSize; vx++) {
        for (let vy = 0; vy < doubleSize; vy++) {
            const lx = vx % size;
            const ly = vy % size;
            const index = ly * size + lx;

            const pointState = boardState[index];
            const cx = padding + vx * cellSize;
            const cy = padding + vy * cellSize;

            const isHovered = hoverX === lx && hoverY === ly && pointState === 0;
            const isLastMove = lastMove && lastMove[0] === lx && lastMove[1] === ly;

            if (aiHeatmap && pointState === 0) {
                const heat = aiHeatmap[index];
                if (heat > 0.01) {
                    const pct = Math.round(heat * 100);
                    const fontSize = Math.max(8, cellSize * 0.32);
                    heatmaps.push(
                        <g key={`heat-${vx}-${vy}`} style={{ pointerEvents: 'none' }}>
                            <circle cx={cx} cy={cy} r={cellSize * 0.45} fill={`rgba(255, 50, 50, ${heat * 0.8})`} />
                            <text
                                x={cx} y={cy}
                                textAnchor="middle" dominantBaseline="central"
                                fill="#fff" fontSize={fontSize} fontWeight="bold"
                                style={{ textShadow: '0 1px 2px rgba(0,0,0,0.6)' }}
                            >{pct}</text>
                        </g>
                    );
                }
            }

            if (pointState !== 0 || isHovered) {
                let fill = 'transparent';
                if (pointState === 1) fill = '#111';
                else if (pointState === -1) fill = '#eee';
                else if (isHovered && currentPlayer === 1) fill = 'rgba(17, 17, 17, 0.4)';
                else if (isHovered && currentPlayer === -1) fill = 'rgba(238, 238, 238, 0.4)';

                stones.push(
                    <g key={`stone-${vx}-${vy}`} style={{ pointerEvents: 'none' }}>
                        <circle cx={cx} cy={cy} r={cellSize * 0.42} fill={fill} stroke={pointState !== 0 ? '#000' : 'none'} strokeWidth={1} />
                        {isLastMove && pointState !== 0 && (
                            <circle cx={cx} cy={cy} r={cellSize * 0.2} stroke={pointState === 1 ? '#fff' : '#000'} strokeWidth={2} fill="none" />
                        )}
                    </g>
                );
            }
        }
    }

    return (
        <svg
            width={containerSize}
            height={containerSize}
            style={{ backgroundColor: '#deae6f', borderRadius: '12px', boxShadow: '0 8px 32px rgba(0,0,0,0.3)', cursor: 'crosshair', touchAction: 'none' }}
            onPointerMove={handlePointerMove}
            onPointerLeave={handlePointerLeave}
            onClick={handleClick}
        >
            {/* Highlight logical center */}
            <rect
                x={highlightX}
                y={highlightY}
                width={highlightSquareWidth}
                height={highlightSquareWidth}
                fill="rgba(255, 255, 255, 0.15)"
                stroke="rgba(255, 255, 255, 0.4)"
                strokeWidth={2}
                strokeDasharray="4 4"
            />
            {lines}
            {heatmaps}
            {stones}
        </svg>
    );
};

export default Board;
