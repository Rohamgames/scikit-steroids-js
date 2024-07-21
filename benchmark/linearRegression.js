// Function to generate random data

import LinearRegression from "../yes/linearModel/LinearRegression.js";

function generateData(samples, features) {
    const X = [];
    const y = [];
    for (let i = 0; i < samples; i++) {
        const row = [];
        for (let j = 0; j < features; j++) {
            row.push(Math.random() * 10);
        }
        X.push(row);
        y.push(Math.random() * 10);
    }
    return { X, y };
}

// Benchmark function
function benchmark(samples, features) {
    const { X, y } = generateData(samples, features);
    const lr = new LinearRegression('gpu');

    console.time('Fit');
    lr.fit(X, y);
    console.timeEnd('Fit');

    console.time('Predict');
    const predictions = lr.predict(X);
    console.timeEnd('Predict');

    console.log(`First 5 Predictions:`, predictions.slice(0, 5));
}

// Example benchmark with large data
benchmark(2, 2); // 10,000 samples, 10 features
