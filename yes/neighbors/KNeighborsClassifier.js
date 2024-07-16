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
            this.distanceKernel = this.gpu.createKernel(function (x, X_train) {
                let distance = 0;
                for (let i = 0; i < this.constants.dim; i++) {
                    distance += Math.pow(Math.abs(x[i] - X_train[this.thread.x][i]), this.constants.p);
                }
                return Math.pow(distance, 1 / this.constants.p);
            }, {
                constants: { dim, p },
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
        console.time('dis')
        const distances = X.map(x => (this.distanceKernel(x, this.X_train)));
        console.timeEnd('dis')

        return X.map((x, idx) => {
            console.time('sort')
            const sortedDistances = distances[idx]
                .map((distance, i) => ({ distance, label: this.y_train[i] }))
                .sort((a, b) => a.distance - b.distance)
                .slice(0, this.n_neighbors);
            console.timeEnd('sort')

            const counts = new Array(this.n_classes).fill(0);
            sortedDistances.forEach(neighbor => {
                const label = neighbor.label;
                counts[label] += 1;
            });

            let maxCount = -1;
            let maxLabel = -1;
            for (let label = 0; label < this.n_classes; label++) {
                if (counts[label] > maxCount) {
                    maxCount = counts[label];
                    maxLabel = label;
                }
            }

            return maxLabel;
        });

    }

    score(X, y) {
        const predictions = this.predict(X);
        const correct = predictions.filter((pred, i) => pred == y[i]).length;
        return correct / y.length;
    }
}
