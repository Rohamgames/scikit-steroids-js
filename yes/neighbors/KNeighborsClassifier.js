import { GPU } from 'gpu.js';

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
        // initialize the kernel now that X_train is available
        const dim = this.X_train[0].length;
        if (this.metric === 'minkowski') {
            const p = this.p;
            this.distanceKernel = this.gpu.createKernel(function (x) {
                let distance = 0;
                for (let i = 0; i < this.constants.dim; i++) {
                    distance += Math.pow(Math.abs(x[i] - this.constants.XT[this.thread.x][i]), this.constants.p);
                }
                return Math.pow(distance, 1 / this.constants.p);
            }, {
                constants: { dim, p, XT: X },
                output: [this.X_train.length]
            });
        } else if (this.metric === 'hamming') {
            this.distanceKernel = this.gpu.createKernel(function (x, X_train) {
                let distance = 0;
                for (let i = 0; i < this.constants.dim; i++) {
                    distance += (x[i] !== X_train[this.thread.x][i] ? 1 : 0);
                }
                return distance / this.constants.dim;
            }, {
                constants: { dim },
                output: [this.X_train.length]
            });
        } else if (this.metric === 'cosine') {
            this.distanceKernel = this.gpu.createKernel(function (x, X_train) {
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
                constants: { dim },
                output: [this.X_train.length]
            });
        }
    }

    predict(X) {
        console.time('DIST')
        const distances = X.map(x => (this.distanceKernel(x)));
        console.timeEnd('DIST')
        return X.map((x, idx) => {
            const indexedDistances = Array.from(distances[idx]).map((distance, i) => ({ distance, label: this.y_train[i], index: i }));
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
