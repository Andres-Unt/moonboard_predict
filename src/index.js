import * as tf from '@tensorflow/tfjs';
import NpyJs from 'npyjs';
import { INPUT_DIMS, POS2I_MAIN, POS2I_START, POS2I_END, GFONT } from './maps.js';

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
export async function loadModel() {
    const model = tf.sequential();

    // Input layer
    model.add(tf.layers.inputLayer({ inputShape: [INPUT_DIMS] }));

    // GaussianNoise layer: only active in training, but we include for structure
    model.add(tf.layers.gaussianNoise({ stddev: 0.05 })); // use your tuned noise_level

    // Hidden blocks: Dense -> BatchNorm -> Dropout
    const numBlocks = 4;    // hp.Choice('num_layers',[4])
    const units = 64;       // hp.Choice('units', [64,96])
    const dropoutRate = 0.1; // hp.Float('dropout',0.1,0.3)
    for (let i = 0; i < numBlocks; i++) {
        model.add(tf.layers.dense({ units, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 1e-6 }) }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: dropoutRate }));
    }

    // Final dense layers
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    // Load weights
    const npy = new NpyJs();
    const layerFiles = {
        1: 2,
        2: 4,
        4: 2,
        5: 4,
        7: 2,
        8: 4,
        10: 2,
        11: 4,
        13: 2,
        14: 2
    };
    for (const [key, count] of Object.entries(layerFiles)) {
        const idx = Number(key);
        const layer = model.layers[idx + 1];
        if (!layer || count === 0) continue;
        const weights = [];
        for (let j = 0; j < count; j++) {
            const { data, shape } = await npy.load(`weights/layer_${idx}_weight_${j}.npy.txt`);
            weights.push(tf.tensor(data, shape, 'float32'));
        }
        layer.setWeights(weights);
        console.log('Loaded layer ' + idx + ' weights:', weights.map(w => w.shape));
    }
    return model;
}

// Run a quick test on page load
export async function runTest() {
    console.log('Initializing model load and test');
    const model = await loadModel();
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
