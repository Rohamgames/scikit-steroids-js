import DecisionTreeClassifier from "../tree/decisionTreeClassifier.js";

export default class RandomForestClassifier {
    constructor(nEstimators = 10) {
        this.nEstimators = nEstimators;
        this.trees = [];
    }

    fit(X, y) {
        const sampleData = (data, labels) => {
            const nSamples = data.length;
            const sampledIndices = Array.from({ length: nSamples }, () => Math.floor(Math.random() * nSamples));
            const sampledData = sampledIndices.map(index => data[index]);
            const sampledLabels = sampledIndices.map(index => labels[index]);
            return { sampledData, sampledLabels };
        };

        for (let i = 0; i < this.nEstimators; i++) {
            const { sampledData, sampledLabels } = sampleData(X, y);
            const tree = new DecisionTreeClassifier();
            tree.fit(sampledData, sampledLabels);
            this.trees.push(tree);
        }
    }

    predict(X) {
        const treePredictions = this.trees.map(tree => tree.predict(X));
        const majorityVote = (predictions) => {
            const counts = {};
            predictions.forEach(pred => counts[pred] = (counts[pred] || 0) + 1);
            return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        };

        const predictions = [];
        for (let i = 0; i < X.length; i++) {
            const rowPredictions = treePredictions.map(treePrediction => treePrediction[i]);
            predictions.push(majorityVote(rowPredictions));
        }

        return predictions;
    }
}
