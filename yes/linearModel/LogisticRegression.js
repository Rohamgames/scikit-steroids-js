export default class LogisticRegression {
    constructor(learningRate = 0.01, iterations = 1000) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weights = null;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    fit(X, y) {
        const m = X.length;
        this.weights = Array(X[0].length).fill(0);

        for (let i = 0; i < this.iterations; i++) {
            const z = math.dot(X, this.weights);
            const h = z.map(this.sigmoid);
            const gradient = math.dot(math.transpose(X), math.subtract(h, y)).map(val => val / m);
            this.weights = math.subtract(this.weights, math.multiply(gradient, this.learningRate));
        }
    }

    predict(X) {
        const z = math.dot(X, this.weights);
        return z.map(this.sigmoid).map(val => val >= 0.5 ? 1 : 0);
    }
}