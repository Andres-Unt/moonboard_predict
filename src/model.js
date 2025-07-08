import * as tf from '@tensorflow/tfjs';
import NpyJs from 'npyjs';
const npy = new NpyJs();

// Build the ConvMoonModel architecture in TF.js
export function buildModel(inputShape) {
    const model = tf.sequential();

    // Input reshape: [H, W, C] expected, so we add a reshape layer
    model.add(tf.layers.inputLayer({ inputShape }));

    // Conv Block helper
    function convBlock(filters, name) {
        model.add(tf.layers.conv2d({
            filters,
            kernelSize: 3,
            padding: 'same',
            useBias: true,
            name: name + '_conv'
        }));
        model.add(tf.layers.batchNormalization({
            axis: -1,
            name: name + '_bn'
        }));
        model.add(tf.layers.activation({ activation: 'relu', name: name + '_act' }));
        model.add(tf.layers.dropout({ rate: parseFloat(name.split('_')[1]) / 10, name: name + '_drop' }));
    }

    convBlock(32, 'block1_1');
    convBlock(64, 'block2_2');
    convBlock(64, 'block3_3');
    convBlock(64, 'block4_4');
    convBlock(64, 'block5_5');

    // Global average pool
    model.add(tf.layers.globalAveragePooling2d({ name: 'gap' }));
    // Final dense to 1 output
    model.add(tf.layers.dense({ units: 1, useBias: true, name: 'fc' }));

    return model;
}

// Async loading of weights
async function loadWeightsForLayer(npy, prefix, modelLayer) {
    // Load kernel and bias
    const k = await npy.load(`weights/${prefix}_weight.npy`);
    const b = await npy.load(`weights/${prefix}_bias.npy`);
    let kernel = tf.tensor(k.data, k.shape, 'float32');
    let bias = tf.tensor(b.data, b.shape, 'float32');
    // For Conv2D in TF.js, weight shape is [kh, kw, in, out]
    // PyTorch uses [out, in, kh, kw]
    if (modelLayer.getClassName() === 'Conv2D') {
        kernel = kernel.transpose([2, 3, 1, 0]);
    }
    modelLayer.setWeights([kernel, bias]);
}

async function loadBatchNorm(prefix, modelLayer) {
    const w = await npy.load(`weights/${prefix}_weight.npy`);
    const b = await npy.load(`weights/${prefix}_bias.npy`);
    const rm = await npy.load(`weights/${prefix}_running_mean.npy`);
    const rv = await npy.load(`weights/${prefix}_running_var.npy`);
    const gamma = tf.tensor(w.data, w.shape, 'float32');
    const beta = tf.tensor(b.data, b.shape, 'float32');
    const mean = tf.tensor(rm.data, rm.shape, 'float32');
    const var_ = tf.tensor(rv.data, rv.shape, 'float32');
    modelLayer.setWeights([gamma, beta, mean, var_]);
}

// Main loader
export async function loadModel() {
    // Input shape: [H, W, C] = [20, 11, 3]
    const model = buildModel([20, 11, 3]);

    // Map layer names to model.layers indices
    const layers = model.layers;

    // conv1 -> layers[1], bn1 -> layers[2], drop -> 4
    await loadWeightsForLayer(npy, 'conv1', layers.find(l => l.name === 'block1_1_conv'));
    await loadBatchNorm('bn1', layers.find(l => l.name === 'block1_1_bn'));
    await loadWeightsForLayer(npy, 'conv2', layers.find(l => l.name === 'block2_2_conv'));
    await loadBatchNorm('bn2', layers.find(l => l.name === 'block2_2_bn'));
    await loadWeightsForLayer(npy, 'conv3', layers.find(l => l.name === 'block3_3_conv'));
    await loadBatchNorm('bn3', layers.find(l => l.name === 'block3_3_bn'));
    await loadWeightsForLayer(npy, 'conv4', layers.find(l => l.name === 'block4_4_conv'));
    await loadBatchNorm('bn4', layers.find(l => l.name === 'block4_4_bn'));
    await loadWeightsForLayer(npy, 'conv5', layers.find(l => l.name === 'block5_5_conv'));
    await loadBatchNorm('bn5', layers.find(l => l.name === 'block5_5_bn'));

    // Fully connected
    const fcLayer = layers.find(l => l.name === 'fc');
    const fck = await npy.load('weights/fc_weight.npy');
    const fcb = await npy.load('weights/fc_bias.npy');
    const fcKernel = tf.tensor(fck.data, fck.shape, 'float32');
    const fcBias = tf.tensor(fcb.data, fcb.shape, 'float32');
    fcLayer.setWeights([fcKernel.transpose(), fcBias]);

    return model;
}

// Example predict
export async function predict(gridTensor) {
    const model = await loadModel();
    return model.predict(gridTensor);
}