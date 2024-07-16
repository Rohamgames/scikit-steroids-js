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
        return X.map(x => Number(this._predictSingle(x)));
    }

    _predictSingle(x) {
        const distances = this._calculateDistances(x);
        const kNearestNeighbors = distances
            .map((distance, i) => ({ distance, label: this.y_train[i] }))
            .sort((a, b) => a.distance - b.distance)
            .slice(0, this.n_neighbors);

        const counts = {};
        kNearestNeighbors.forEach(neighbor => {
            const label = neighbor.label;
            counts[label] = (counts[label] || 0) + 1;
        });

        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }


    _calculateDistance(a, b) {
        switch (this.metric) {
            case 'minkowski':
                return this._minkowskiDistance(a, b, this.p);
            case 'hamming':
                return this._hammingDistance(a, b);
            case 'cosine':
                return this._cosineDistance(a, b);
            default:
                throw new Error(`Unknown metric: ${this.metric}`);
        }
    }

    _minkowskiDistance(a, b, p) {
        return Math.pow(a.reduce((sum, a_i, i) => sum + Math.abs(a_i - b[i]) ** p, 0), 1 / p);
    }

    _hammingDistance(a, b) {
        return a.reduce((sum, a_i, i) => sum + (a_i !== b[i] ? 1 : 0), 0) / a.length;
    }

    _cosineDistance(a, b) {
        const dotProduct = a.reduce((sum, a_i, i) => sum + a_i * b[i], 0);
        const magnitudeA = Math.sqrt(a.reduce((sum, a_i) => sum + a_i ** 2, 0));
        const magnitudeB = Math.sqrt(b.reduce((sum, b_i) => sum + b_i ** 2, 0));
        return 1 - (dotProduct / (magnitudeA * magnitudeB));
    }

    score(X, y) {
        const predictions = this.predict(X);
        const correct = predictions.filter((pred, i) => pred == y[i]).length;
        return correct / y.length;
    }
}
