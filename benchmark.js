import KNeighborsClassifier from "./yes/neighbors/KNeighborsClassifier.js";

import trainTestSplit from "./yes/modelSelection/trainTestSplit.js";
import nj  from 'numjs';
import accuracyScore from "./yes/metrics/accuracyScore.js";

console.time('Time for data generation');
const numDataPoints = 1024*1024;
const numFeatures = 128;

const randomData = Array.from({ length: numDataPoints }, () =>
    Array.from({ length: numFeatures }, () => Math.random())
);

// Generate random labels (rounded to nearest integer)
const randomLabels = Array.from({ length: numDataPoints }, () =>
    Math.round(Math.random())
);
console.timeEnd('Time for data generation');

console.time('Time for data split');
let { X_train, X_test, y_train, y_test } = trainTestSplit(randomData, randomLabels, 0.001);
console.timeEnd('Time for data split');

const pValues = [1, 1.5, 2, 3];

pValues.forEach(p => {
    console.time('knn def');
    const knn = new KNeighborsClassifier(5, 'minkowski', p, 'gpu');
    knn.fit(X_train, y_train);
    console.timeEnd('knn def')
    console.time('knn pred')
    const y_pred = knn.predict(X_test);
    console.timeEnd('knn pred')
    console.time('acc')
    const accuracy = accuracyScore(y_test, y_pred);
    console.timeEnd('acc')
    console.log(`Accuracy with GPU Minkowski distance (p=${p}): ${accuracy}`);
});

//
// const X_train = [[1, 2], [2, 3], [3, 4], [6, 7]];
// const y_train = [0, 0, 0, 1];
//
// const X_test = [[1, 2], [5, 5]];
// const y_test = [0, 1];
//
// // Initialize KNN with Minkowski distance and p=2
// const knnMinkowski = new KNeighborsClassifier(1, 'minkowski', 2, 'cpu');
//
// // Fit the model
// knnMinkowski.fit(X_train, y_train);
//
// // Predict
// const predictionsMinkowski = knnMinkowski.predict(X_test);
// console.log(predictionsMinkowski); // Output predictions for X_test
//
// // Score the model
// const accuracyMinkowski = accuracyScore(y_test,predictionsMinkowski);
// console.log(accuracyMinkowski); // Output accuracy score
//


pValues.forEach(p => {
    const start = performance.now();
    const knn = new KNeighborsClassifier(5, 'minkowski', p, 'cpu');
    knn.fit(X_train, y_train);
    const y_pred = knn.predict(X_test);
    const accuracy = accuracyScore(y_test, y_pred);
    const end = performance.now();
    console.log(`Accuracy with CPU Minkowski distance (p=${p}): ${accuracy}`);
    console.log(`Time taken: ${(end - start) / 1000} seconds`);
});



//
// // Initialize KNN with Minkowski distance and p=3
// const p = 2;
// const metric = 'minkowski';
// const k = 500;
// const knnCpu = new KNeighborsClassifier(k, metric, p, 'cpu');
// // Benchmarking

// // console.time('Training Time ');
// knnCpu.fit(X_train, y_train);
// // console.timeEnd('Training Time');
//
// console.time('Prediction Time CPU');
// knnCpu.predict(X_test);
// console.timeEnd('Prediction Time CPU');
//
//
// const knnGpu = new KNeighborsClassifier(k, metric, p, 'gpu');
// // Benchmarking
// // console.time('Training Time');
// knnGpu.fit(X_train, y_train);
// // console.timeEnd('Training Time');
//
// console.time('Prediction Time GPU');
// knnGpu.predict(X_test);
// console.timeEnd('Prediction Time GPU');
//
// // console.time('Scoring Time');
// // const accuracy = knn.score(X_test, y_test);
// // console.timeEnd('Scoring Time');
//
// // console.log('Accuracy:', accuracy);
//
//
// //
// // const X_train = [[1, 2], [2, 3], [3, 4], [6, 7]];
// // const y_train = [0, 0, 0, 1];
// //
// // // Test
// // const X_test = [[1, 2], [5, 5]];
// // const y_test = [0, 1];
// //
// // // Initialize KNN with Minkowski distance and p=2
// // const knnMinkowski = new KNeighborsClassifier(3, 'minkowski', 2);
// //
// // // Fit the model
// // knnMinkowski.fit(X_train, y_train);
// //
// // // Predict
// // const predictionsMinkowski = knnMinkowski.predict(X_test);
// // console.log(predictionsMinkowski); // Output predictions for X_test
// //
// // // Score the model
// // const accuracyMinkowski = knnMinkowski.score(X_test, y_test);
// // console.log(accuracyMinkowski); // Output accuracy score
// //
// // // Initialize KNN with Hamming distance
// // const knnHamming = new KNeighborsClassifier(3, 'hamming');
// //
// // // Fit the model
// // knnHamming.fit(X_train, y_train);
// //
// // // Predict
// // const predictionsHamming = knnHamming.predict(X_test);
// // console.log(predictionsHamming); // Output predictions for X_test
// //
// // // Score the model
// // const accuracyHamming = knnHamming.score(X_test, y_test);
// // console.log(accuracyHamming); // Output accuracy score
// //
// // // Initialize KNN with Cosine distance
// // const knnCosine = new KNeighborsClassifier(3, 'cosine');
// //
// // // Fit the model
// // knnCosine.fit(X_train, y_train);
// //
// // // Predict
// // const predictionsCosine = knnCosine.predict(X_test);
// // console.log(predictionsCosine); // Output predictions for X_test
// //
// // // Score the model
// // const accuracyCosine = knnCosine.score(X_test, y_test);
// // console.log(accuracyCosine); // Output accuracy score
// //
// //
// //
// //
