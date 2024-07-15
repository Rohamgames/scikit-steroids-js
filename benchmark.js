import KNeighborsClassifier from "./yes/neighbors/KNeighborsClassifier.js";
function generateRandomData(numSamples, numFeatures) {
    const X = [];
    const y = [];
    for (let i = 0; i < numSamples; i++) {
        const sample = [];
        for (let j = 0; j < numFeatures; j++) {
            sample.push(Math.random());
        }
        X.push(sample);
        y.push(Math.floor(Math.random() * 2)); // Binary classification (0 or 1)
    }
    return { X, y };
}

// Generate large dataset
const { X: X_train, y: y_train } = generateRandomData(10000, 50); // 10,000 samples, 50 features each
const { X: X_test, y: y_test } = generateRandomData(1000, 50); // 1,000 samples, 50 features each

// Initialize KNN with Minkowski distance and p=3

const knn = new KNeighborsClassifier(3, 'minkowski', 2, 'parallel');
// Benchmarking
console.time('Training Time');
knn.fit(X_train, y_train);
console.timeEnd('Training Time');

console.time('Prediction Time');
const predictions = knn.predict(X_test);
console.timeEnd('Prediction Time');

console.time('Scoring Time');
const accuracy = knn.score(X_test, y_test);
console.timeEnd('Scoring Time');

console.log('Accuracy:', accuracy);


//
// const X_train = [[1, 2], [2, 3], [3, 4], [6, 7]];
// const y_train = [0, 0, 0, 1];
//
// // Test
// const X_test = [[1, 2], [5, 5]];
// const y_test = [0, 1];
//
// // Initialize KNN with Minkowski distance and p=2
// const knnMinkowski = new KNeighborsClassifier(3, 'minkowski', 2);
//
// // Fit the model
// knnMinkowski.fit(X_train, y_train);
//
// // Predict
// const predictionsMinkowski = knnMinkowski.predict(X_test);
// console.log(predictionsMinkowski); // Output predictions for X_test
//
// // Score the model
// const accuracyMinkowski = knnMinkowski.score(X_test, y_test);
// console.log(accuracyMinkowski); // Output accuracy score
//
// // Initialize KNN with Hamming distance
// const knnHamming = new KNeighborsClassifier(3, 'hamming');
//
// // Fit the model
// knnHamming.fit(X_train, y_train);
//
// // Predict
// const predictionsHamming = knnHamming.predict(X_test);
// console.log(predictionsHamming); // Output predictions for X_test
//
// // Score the model
// const accuracyHamming = knnHamming.score(X_test, y_test);
// console.log(accuracyHamming); // Output accuracy score
//
// // Initialize KNN with Cosine distance
// const knnCosine = new KNeighborsClassifier(3, 'cosine');
//
// // Fit the model
// knnCosine.fit(X_train, y_train);
//
// // Predict
// const predictionsCosine = knnCosine.predict(X_test);
// console.log(predictionsCosine); // Output predictions for X_test
//
// // Score the model
// const accuracyCosine = knnCosine.score(X_test, y_test);
// console.log(accuracyCosine); // Output accuracy score
//
//
//
//
