const fs = require('fs');

const boardSize = 4;
const totalPoints = boardSize * boardSize;

const dummyHeatmap = new Array(totalPoints).fill(0).map(() => Math.random());
// Normalize so max is 1 for better visual contrast
const max = Math.max(...dummyHeatmap);
const normalized = dummyHeatmap.map(v => v / max);

const aiNetwork = {
    description: "Dummy 4x4 AI Torus Go Evaluation",
    boardSize: boardSize,
    heatmap: normalized,
    value: Math.random() * 2 - 1 // between -1 and 1
};

fs.writeFileSync('dummy-4x4-ai.json', JSON.stringify(aiNetwork, null, 2));

console.log('Successfully generated dummy-4x4-ai.json');
