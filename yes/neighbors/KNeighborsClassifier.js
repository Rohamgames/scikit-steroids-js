import { GPU } from 'gpu.js';

export default class KNeighborsClassifier {
    constructor(n_neighbors = 5, metric = 'minkowski', p = 2, mode='cpu') {
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
    }

    predict(X) {
        const minkowskiKernel = this.gpu.createKernel(function(x, X_train, p) {
            let distance = 0;
            for (let i = 0; i < this.constants.dim; i++) {
                distance += Math.pow(Math.abs(x[i] - X_train[this.thread.x][i]), p);
            }
            return Math.pow(distance, 1 / p);
        }, {
            constants: { dim: this.X_train[0].length },
            output: [this.X_train.length]
        });

        const hammingKernel = this.gpu.createKernel(function(x, X_train) {
            let distance = 0;
            for (let i = 0; i < this.constants.dim; i++) {
                distance += (x[i] !== X_train[this.thread.x][i] ? 1 : 0);
            }
            return distance / this.constants.dim;
        }, {
            constants: { dim: this.X_train[0].length },
            output: [this.X_train.length]
        });

        const cosineKernel = this.gpu.createKernel(function(x, X_train) {
            let dotProduct = 0;
            let magnitudeX = 0;
            let magnitudeTrain = 0;
            for (let i = 0; i < this.constants.dim; i++) {
                dotProduct += x[i] * X_train[this.thread.x][i];
                magnitudeX += x[i] * x[i];
                magnitudeTrain += X_train[this.thread.x][i] * X_train[this.thread.x][i];
            }
            return 1 - (dotProduct / (Math.sqrt(magnitudeX) * Math.sqrt(magnitudeTrain)));
        }, {
            constants: { dim: this.X_train[0].length },
            output: [this.X_train.length]
        });

        const getDistances = (x) => {
            if (this.metric === 'minkowski') {
                return minkowskiKernel(x, this.X_train, this.p);
            } else if (this.metric === 'hamming') {
                return hammingKernel(x, this.X_train);
            } else if (this.metric === 'cosine') {
                return cosineKernel(x, this.X_train);
            }
        };

        return X.map(x => {
            const distances = getDistances(x);
            const distanceArray = Array.from(distances);
            const kNearestNeighbors = distanceArray
                .map((distance, i) => ({ distance, label: this.y_train[i] }))
                .sort((a, b) => a.distance - b.distance)
                .slice(0, this.n_neighbors);
            const counts = {};

            kNearestNeighbors.forEach(neighbor => {
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
