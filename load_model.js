import * as tf from '@tensorflow/tfjs';
import NpyJs from 'npyjs';

// your grade labels:
export const GFONT = [
  '6A+','6B','6B+','6C','6C+',
  '7A','7A+','7B','7B+','7C',
  '7C+','8A','8A+','8B','8B+',
];

// position → index maps (just copy your Python dicts)
export const POS2I_MAIN = { /* … */ };
export const POS2I_START = { /* … */ };
export const POS2I_END  = { /* … */ };

function encodeHolds(holds) {
  const vecMain = new Array(Object.keys(POS2I_MAIN).length).fill(0);
  const vecStart = new Array(Object.keys(POS2I_START).length).fill(0);
  const vecEnd  = new Array(Object.keys(POS2I_END).length).fill(0);

  for (let raw of holds) {
    let h = raw.trim().toUpperCase();
    let isStart = h.endsWith('S');
    let isEnd   = h.endsWith('T');
    if (isStart || isEnd) h = h.slice(0, -1);

    if (h in POS2I_MAIN) vecMain[POS2I_MAIN[h]] = 1;
    if (isStart && h in POS2I_START) vecStart[POS2I_START[h]] = 1;
    if (isEnd   && h in POS2I_END)   vecEnd[ POS2I_END[h] ] = 1;
  }

  // tfjs wants a 2D tensor [batch, features]
  return tf.tensor2d([ vecMain.concat(vecStart).concat(vecEnd) ]);
}

async function loadWeightsForLayer(model, layerIndex, npy) {
  const layer = model.layers[layerIndex];
  // each layer may have 0, 1 or 2 weight arrays (kernel, bias), or 4 for BatchNorm
  const weightFiles = [];
  for (let i = 0; ; i++) {
    const url = `weights/layer_${layerIndex}_weight_${i}.npy`;
    try {
      await fetch(url, { method: 'HEAD' }); // check exists
      weightFiles.push(url);
    } catch {
      break;
    }
  }
  // load all in parallel
  const arrays = await Promise.all(weightFiles.map(url => npy.load(url)));
  // convert to tf tensors
  const tensors = arrays.map(obj =>
    tf.tensor(obj.data, obj.shape, 'float32')
  );
  layer.setWeights(tensors);
}

export async function loadModel() {
  // 1) Re‑build the architecture
  const model = tf.sequential();
  // (1) Input layer
  model.add(tf.layers.inputLayer({ inputShape: [ modelJson.config.layers[0].config.batch_input_shape[1] ] }));
  // (2) GaussianNoise was only during training—skip at inference
  // (3) Dense+BatchNorm+Dropout blocks
  const numDenseBlocks = /* count how many hp.Choice('num_layers',[4]) ended up going through */;
  for (let i = 0; i < numDenseBlocks; i++) {
    model.add(tf.layers.dense({ units: /* your chosen hp.units */, activation: 'relu' }));
    model.add(tf.layers.batchNormalization());
    // Dropout is no‑op at inference so you can skip adding it
  }
  // final Dense(16) + Dense(1)
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

  // 2) Load all weights
  const npy = new NpyJs();
  for (let i = 0; i < model.layers.length; i++) {
    await loadWeightsForLayer(model, i, npy);
  }

  return model;
}

export async function predictGrade(holds) {
  const model = await loadModel();
  const x = encodeHolds(holds);
  const p = model.predict(x).arraySync()[0][0];
  const idx = Math.round(Math.min(Math.max(p, 0), GFONT.length - 1));
  return { numeric: p, label: GFONT[idx] };
}

