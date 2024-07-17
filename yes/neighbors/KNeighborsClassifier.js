// import { GPU } from 'gpu.js';

export default class KNeighborsClassifier {
    constructor(n_neighbors = 5, metric = 'minkowski', p = 2, mode = 'cpu') {
        this.n_neighbors = n_neighbors;
        this.metric = metric;
        this.p = p;
        this.X_train = null;
        this.y_train = null;

        this.gpu = new GPU({ mode });
    }

    fit(X, y) {
        this.X_train = X;
        this.y_train = y.map(label => Number(label));  // Ensure y_train contains numbers

        let maxClass = -1;
        for (let i = 0; i < y.length; i++) {
            if (y[i] > maxClass) {
                maxClass = y[i];
            }
        }
        const numClasses = maxClass + 1;
        this.n_classes = numClasses;

        const dim = this.X_train[0].length;

        if (this.metric === 'minkowski') {
            const p = this.p;
            if(p===2){
                this.indexDistanceKernel = this.gpu.createKernel(function (x) {
                    let distance = 0;
                    for (let i = 0; i < this.constants.dim; i++) {
                        let diff = x[i] - this.constants.X_train[this.thread.x][i];
                        distance += (diff ** 2);
                    }
                    return [Math.sqrt(distance), this.thread.x];
                }, {
                    constants: {dim, X_train: this.X_train},
                    output: [this.X_train.length],
                    pipeline: true
                });
            }else {
                this.indexDistanceKernel = this.gpu.createKernel(function (x) {
                    let distance = 0;
                    for (let i = 0; i < this.constants.dim; i++) {
                        distance += Math.pow(Math.abs(x[i] - this.constants.X_train[this.thread.x][i]), this.constants.p);
                    }
                    return [Math.pow(distance, 1 / this.constants.p), this.thread.x];
                }, {
                    constants: {dim, p, X_train: this.X_train},
                    output: [this.X_train.length],
                    pipeline: true
                });
            }
        }
    }

    predict(X) {
        const distances = X.map(x => Array.from(this.indexDistanceKernel(x)));
        return distances.map((d) => {
            const indexedDistances = d.map((value) => ({ distance: value[0], index: value[1], label: this.y_train[value[1]] }));
            indexedDistances.sort((a, b) => a.distance - b.distance);
            const nearestNeighbors = indexedDistances.slice(0, this.n_neighbors);
            const counts = new Array(this.n_classes).fill(0);
            nearestNeighbors.forEach(neighbor => {
                const label = neighbor.label;
                counts[label] = (counts[label] || 0) + 1;
            });

            return Number(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b));
        });
    }

    score(X, y) {
        const predictions = this.predict(X);
        const correct = predictions.filter((pred, i) => pred == y[i]).length;
        return correct / y.length;
    }
}