// require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k) {
    return features
        .sub(predictionPoint)
        .pow(2)
        .sum(1, { keepDims: true })
        .pow(0.5)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => {
            a.slice(0, 1).dataSync()[0] > b.slice(0, 1).dataSync()[0] ? 1: -1;
        })
        .slice(0, k)
        .reduce((acc ,pair) => acc + pair.slice(1, 1).dataSync()[0], 0) / k;
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);
// testFeatures = tf.tensor(testFeatures);
// testLabels = tf.tensor(testLabels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0]
    console.log('Error', err * 100);
});
