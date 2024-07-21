import KNeighborsClassifier from "./yes/neighbors/KNeighborsClassifier.js";
import trainTestSplit from "./yes/modelSelection/trainTestSplit.js";
import accuracyScore from "./yes/metrics/accuracyScore.js";

const numFeatures = 128;
const pValues = [1, 1.5, 2, 3];
const dataPointsToTest = [1024, 1024*2, 1024*4, 1024*8, 1024*16, 1024*32, 1024*64];

// Prepare an array to store results
let results = [];

// Function to run the KNN process and capture timing data
async function runKNN(numDataPoints) {
    console.log(`Running for ${numDataPoints} data points...`);

    const randomData = Array.from({ length: numDataPoints }, () =>
        Array.from({ length: numFeatures }, () => Math.random())
    );

    const randomLabels = Array.from({ length: numDataPoints }, () =>
        Math.round(Math.random())
    );

    let { X_train, X_test, y_train, y_test } = trainTestSplit(randomData, randomLabels, 0.01);

    let timings = [];

    // GPU Minkowski distance
    for (let p of pValues) {
        const start = Date.now()
        const knnGPU = new KNeighborsClassifier(5, 'minkowski', p, 'cpu');
        let accuracy_gpu=0;
        for(let i=0; i<5; i++) {
            await knnGPU.fit(X_train, y_train);
            const y_pred_gpu = knnGPU.predict(X_test);
            accuracy_gpu += accuracyScore(y_test, y_pred_gpu);
        }
        timings.push({
            method: 'CPU',
            p: p,
            time: (Date.now() - start)/5,
            accuracy: accuracy_gpu/5,
            numDataPoints: numDataPoints
        });
    }

    // // CPU Minkowski distance
    // for (let p of pValues) {
    //     const start = Date.now()
    //     const knnCPU = new KNeighborsClassifier(5, 'minkowski', p, 'cpu');
    //     console.time(`CPU (p=${p})`);
    //     await knnCPU.fit(X_train, y_train);
    //     const y_pred_cpu = knnCPU.predict(X_test);
    //     const accuracy_cpu = accuracyScore(y_test, y_pred_cpu);
    //     console.timeEnd(`CPU (p=${p})`);
    //     timings.push({
    //         method: 'CPU',
    //         p: p,
    //         time: Date.now() - start,
    //         accuracy: accuracy_cpu,
    //         numDataPoints: numDataPoints
    //     });
    // }

    results.push(...timings);
}

// Run for each specified number of data points
async function runForDataPoints() {
    for (let numDataPoints of dataPointsToTest) {
        await runKNN(numDataPoints);
    }

    // Output results to console for easy copy to Excel or CSV
    console.log("Results:");
    console.table(results);
}
alert('press to start')
runForDataPoints();
