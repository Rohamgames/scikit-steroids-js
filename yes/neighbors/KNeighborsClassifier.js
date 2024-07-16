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
        const distancesKernel = this.gpu.createKernel(function(x, X_train, metric, p) {
            let distance = 0;
            for (let i = 0; i < this.constants.dim; i++) {
                if (metric === 0) {  // minkowski
                    distance += Math.pow(Math.abs(x[i] - X_train[this.thread.x][i]), p);
                } else if (metric === 1) {  // hamming
                    distance += (x[i] !== X_train[this.thread.x][i] ? 1 : 0);
                } else if (metric === 2) {  // cosine
                    distance += x[i] * X_train[this.thread.x][i];
                }
            }
            if (metric === 0) {  // minkowski
                return Math.pow(distance, 1 / p);
            } else if (metric === 1) {  // hamming
                return distance / this.constants.dim;
            } else if (metric === 2) {  // cosine
                let magnitudeX = 0, magnitudeTrain = 0;
                for (let i = 0; i < this.constants.dim; i++) {
                    magnitudeX += x[i] * x[i];
                    magnitudeTrain += X_train[this.thread.x][i] * X_train[this.thread.x][i];
                }
                return 1 - (distance / (Math.sqrt(magnitudeX) * Math.sqrt(magnitudeTrain)));
            }
        }, {
            constants: { dim: this.X_train[0].length },
            output: [this.X_train.length]
        });

        return X.map(x => {
            const distances = distancesKernel(x, this.X_train, this.metric === 'minkowski' ? 0 : this.metric === 'hamming' ? 1 : 2, this.p);
            const kNearestNeighbors = Array.from(distances)
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
