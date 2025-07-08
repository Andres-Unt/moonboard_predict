import * as tf from '@tensorflow/tfjs';
import NpyJs from 'npyjs';
import { POS2I_MAIN, POS2I_START, POS2I_END, GFONT } from './maps.js';
const npy = new NpyJs();

// 1) Build the ConvMoonModel architecture in TF.js
export function buildModel() {
    const model = tf.sequential();

    // Expect input shape [20,11,3]
    model.add(tf.layers.inputLayer({ inputShape: [20, 11, 3] }));

    // Helper to add Conv -> BN -> ReLU -> Dropout
    function addBlock(filters, name, dropRate) {
        model.add(tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', useBias: true, name: `${name}_conv` }));
        model.add(tf.layers.batchNormalization({ axis: -1, name: `${name}_bn` }));
        model.add(tf.layers.activation({ activation: 'relu', name: `${name}_act` }));
        model.add(tf.layers.dropout({ rate: dropRate, name: `${name}_drop` }));
    }

    addBlock(32, 'conv1', 0.1);
    addBlock(64, 'conv2', 0.2);
    addBlock(64, 'conv3', 0.3);
    addBlock(64, 'conv4', 0.4);
    addBlock(64, 'conv5', 0.5);

    model.add(tf.layers.globalAveragePooling2d({ name: 'gap' }));
    model.add(tf.layers.dense({ units: 1, useBias: true, name: 'fc' }));

    return model;
}

// 2) Load weights for a given layer
async function loadConv(prefix, layer) {
    const k = await npy.load(`weights/${prefix}_weight.npy.txt`);
    const b = await npy.load(`weights/${prefix}_bias.npy.txt`);
    let W = tf.tensor(k.data, k.shape, 'float32');
    const B = tf.tensor(b.data, b.shape, 'float32');
    // PyTorch: [out, in, kh, kw] -> TF.js: [kh, kw, in, out]
    W = W.transpose([2, 3, 1, 0]);
    layer.setWeights([W, B]);
}
async function loadBN(prefix, layer) {
    const w = await npy.load(`weights/${prefix}_weight.npy.txt`);
    const b = await npy.load(`weights/${prefix}_bias.npy.txt`);
    const rm = await npy.load(`weights/${prefix}_running_mean.npy.txt`);
    const rv = await npy.load(`weights/${prefix}_running_var.npy.txt`);
    const gamma = tf.tensor(w.data, w.shape, 'float32');
    const beta = tf.tensor(b.data, b.shape, 'float32');
    const mean = tf.tensor(rm.data, rm.shape, 'float32');
    const var_ = tf.tensor(rv.data, rv.shape, 'float32');
    layer.setWeights([gamma, beta, mean, var_]);
}
async function loadFC(prefix, layer) {
    const k = await npy.load(`weights/${prefix}_weight.npy.txt`);
    const b = await npy.load(`weights/${prefix}_bias.npy.txt`);
    // k.shape = [out, in]
    const W = tf.tensor(k.data, k.shape, 'float32').transpose();
    const B = tf.tensor(b.data, b.shape, 'float32');
    layer.setWeights([W, B]);
}

// 3) Load full model
export async function loadModel() {
    const model = buildModel();
    const L = model.layers;

    await loadConv('conv1', L.find(l => l.name === 'conv1_conv'));
    await loadBN('bn1', L.find(l => l.name === 'conv1_bn'));
    await loadConv('conv2', L.find(l => l.name === 'conv2_conv'));
    await loadBN('bn2', L.find(l => l.name === 'conv2_bn'));
    await loadConv('conv3', L.find(l => l.name === 'conv3_conv'));
    await loadBN('bn3', L.find(l => l.name === 'conv3_bn'));
    await loadConv('conv4', L.find(l => l.name === 'conv4_conv'));
    await loadBN('bn4', L.find(l => l.name === 'conv4_bn'));
    await loadConv('conv5', L.find(l => l.name === 'conv5_conv'));
    await loadBN('bn5', L.find(l => l.name === 'conv5_bn'));

    await loadFC('fc', L.find(l => l.name === 'fc'));
    return model;
}

// 4) Encode holds into a [1,20,11,3] tensor
export function encodeHolds(holds) {
    const rows = ['-1', '0', ...Array.from({ length: 18 }, (_, i) => String(i + 1))];
    const cols = Array.from({ length: 11 }, (_, i) => String.fromCharCode(65 + i));
    const fh = ['B-1', 'D-1', 'F-1', 'H-1', 'J-1', 'B0', 'D0', 'F0', 'H0', 'J0'];
    const main = new Array(rows.length * cols.length).fill(0);
    const start = new Array(main.length).fill(0);
    const end = new Array(main.length).fill(0);

    for (let raw of holds) {
        let h = raw.trim().toUpperCase();
        const isStart = h.endsWith('S');
        const isEnd = h.endsWith('T');
        if (isStart || isEnd) h = h.slice(0, -1);
        const idx = cols.indexOf(h[0]) + rows.indexOf(h.slice(1)) * cols.length;
        if (idx >= 0 && idx < main.length) {
            main[idx] = 1;
            if (isStart) start[idx] = 1;
            if (isEnd) end[idx] = 1;
        }
    }
    // ensure footholds present
    for (let p of fh) {
        const c = p[0]; const r = p.slice(1);
        const i = cols.indexOf(c) + rows.indexOf(r) * cols.length;
        main[i] = 1;
    }
    // stack and reshape to [1,20,11,3]
    const combined = tf.tensor(main.concat(start, end), [1, 3, rows.length, cols.length]);
    // TF.js expects NHWC
    return combined.transpose([0, 2, 3, 1]);
}

// 5) Predict grade label
export async function predictGrade(holds) {
    const model = await loadModel();
    const x = encodeHolds(holds);
    const p = (await model.predict(x).array())[0][0];
    const idx = Math.round(Math.min(Math.max(p, 0), GFONT.length - 1));
    return GFONT[idx];
}

// Example usage:
// (async ()=> console.log(await predictGrade(['C16','D18T','E13','F18T','F5S','F8','H10','H5S','I7','J11','K4'])) )();
// Example usage:

(async () => console.log(await predictGrade(['C16', 'D18T', 'E13', 'F18T', 'F5S', 'F8', 'H10', 'H5S', 'I7', 'J11', 'K4'])))();
import { initUI } from './ui.js';
initUI();
