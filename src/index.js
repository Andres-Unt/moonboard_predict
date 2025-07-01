import * as tf from '@tensorflow/tfjs';
import NpyJs from 'npyjs';
import { INPUT_DIMS, POS2I_MAIN, POS2I_START, POS2I_END, GFONT } from './maps.js';
const npy = new NpyJs();

// Encode the holds list into a 2D tensor [1, features]
export function encodeHolds(holds) {
    var lenMain = Object.keys(POS2I_MAIN).length;
    var lenStart = Object.keys(POS2I_START).length;
    var lenEnd = Object.keys(POS2I_END).length;
    const vecMain = new Array(lenMain).fill(0);
    const vecStart = new Array(lenStart).fill(0);
    const vecEnd = new Array(lenEnd).fill(0);

    for (const raw of holds) {
        let h = raw.trim().toUpperCase();
        const isStart = h.endsWith('S');
        const isEnd = h.endsWith('T');
        if (isStart || isEnd) h = h.slice(0, -1);
        if (h in POS2I_MAIN) vecMain[POS2I_MAIN[h]] = 1;
        if (isStart && (h in POS2I_START)) vecStart[POS2I_START[h]] = 1;
        if (isEnd && (h in POS2I_END)) vecEnd[POS2I_END[h]] = 1;
    }

    console.log('end', POS2I_END)
    console.log('vecMain:', vecMain);
    console.log('vecStart:', vecStart)
    console.log('vecEnd:', vecEnd);
    console.log('vecmain length:', vecMain.length);
    console.log('vecstart length:', vecStart.length);
    console.log('vecend length:', vecEnd.length);
    const combined = vecMain.concat(vecStart, vecEnd);
    return tf.tensor2d([combined]);
}

// Build the model architecture exactly as in Python (with GaussianNoise, BatchNorm, Dropout)

export async function loadModel(inputDim) {
    console.log('Loading model with input dimension:', inputDim);
    const model = tf.sequential();

    // 1) Input layer
    model.add(tf.layers.inputLayer({ inputShape: [inputDim] }));

    // 2) Gaussian noise layer (only active in training)
    model.add(tf.layers.gaussianNoise({ stddev: 0.05 }));

    // 3) First Linear block: Dense(512) -> ReLU -> Dropout(0.3)
    model.add(tf.layers.dense({ units: 512, useBias: true }));
    model.add(tf.layers.activation({ activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));

    // 4) Second Linear block: Dense(512) -> ReLU -> Dropout(0.3)
    model.add(tf.layers.dense({ units: 512, useBias: true }));
    model.add(tf.layers.activation({ activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));

    // 5) Linear(512 -> 15)
    model.add(tf.layers.dense({ units: 15, useBias: true }));

    // 6) Linear(15 -> 1)
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    // Now load weights from files named like layer_0_weight_0.npy.txt etc.

    // Layers with weights: dense layers only (indexes 2,5,8,10)
    // TF.js layers array includes all layers: inputLayer=0, gaussianNoise=1, dense=2, activation=3, dropout=4, ...
    // Dense layers in this model are at indexes: 2, 5, 8, 10

    const denseLayerIndices = [2, 5, 8, 9];
    console.log('Dense layer indices:', denseLayerIndices);

    for (let i = 0; i < denseLayerIndices.length; i++) {
        const layerIndex = denseLayerIndices[i];
        const layer = model.layers[layerIndex];

        // Load weight and bias files for this layer
        // Files: layer_{i}_weight_0.npy.txt and layer_{i}_bias_1.npy.txt
        const weightFile = `weights/layer_${i}_weight_0.npy.txt`;
        const biasFile = `weights/layer_${i}_bias_1.npy.txt`;

        const weightData = await loadNpyTensor(weightFile);
        const biasData = await loadNpyTensor(biasFile);
        console.log(`Layer ${i} weight shape:`, weightData.shape);
        console.log(`Layer ${i} bias shape:`, biasData.shape);

        // Convert loaded data to tensors with correct shapes
        const weightTensor = tf.tensor(weightData.data, weightData.shape, 'float32').transpose();
        const biasTensor = tf.tensor(biasData.data, biasData.shape, 'float32');


        // Set weights on the layer: [kernel, bias]
        console.log(`Setting weights for layer ${i} (TF layer index ${layerIndex})`);
        layer.setWeights([weightTensor, biasTensor]);

        console.log(`Loaded weights for layer ${i} (TF layer index ${layerIndex}): weight shape ${weightTensor.shape}, bias shape ${biasTensor.shape}`);
    }

    return model;
}


async function loadNpyTensor(file) {
    return await npy.load(file);
    const response = await fetch(file);
    console.log('got response for file:', file, response);
    const buffer = await response.arrayBuffer();
    const { data, shape } = await npy.parse(buffer);
    if (!shape.every(dim => Number.isInteger(dim) && dim > 0)) {
        throw new Error(`Invalid shape from ${file}: ${shape}`);
    }
    return tf.tensor(data, shape, 'float32');
}


// Run a quick test on page load
export async function runTest() {
    console.log('Initializing model load and test');
    const model = await loadModel(275);
    console.log('Model loaded');
    const sample = ['C16', 'D18t', 'E13', 'F18T', 'F5s', 'f8', 'H10', 'h5s', 'i7', 'j11', 'K4'];
    const x = encodeHolds(sample);
    console.log('Input tensor shape:', x.shape);
    const p = (await model.predict(x).array())[0][0];
    const idx = Math.round(Math.min(Math.max(p, 0), GFONT.length - 1));
    console.log('Prediction raw:', p);
    console.log('Prediction label:', GFONT[idx]);
}

runTest();
import { initUI } from './ui.js';
initUI();
