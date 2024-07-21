export default class DecisionTreeClassifier {
    constructor() {
        this.tree = null;
    }

    fit(X, y) {
        const buildTree = (data, labels) => {
            if (labels.every(val => val === labels[0])) return labels[0];
            if (data[0].length === 0) return this.majorityVote(labels);

            const { feature, threshold } = this.bestSplit(data, labels);
            const leftIndices = data.map(row => row[feature] <= threshold);
            const rightIndices = data.map(row => row[feature] > threshold);

            return {
                feature,
                threshold,
                left: buildTree(this.subset(data, leftIndices), this.subset(labels, leftIndices)),
                right: buildTree(this.subset(data, rightIndices), this.subset(labels, rightIndices)),
            };
        };

        this.tree = buildTree(X, y);
    }

    bestSplit(data, labels) {
        let bestFeature = null;
        let bestThreshold = null;
        let bestGini = 1;

        const gini = (groups, classes) => {
            const nInstances = groups.reduce((acc, group) => acc + group.length, 0);
            return groups.reduce((acc, group) => {
                const size = group.length;
                if (size === 0) return acc;
                const score = classes.reduce((scoreAcc, classVal) => {
                    const proportion = group.filter(val => val === classVal).length / size;
                    return scoreAcc + proportion * proportion;
                }, 0);
                return acc + (1 - score) * (size / nInstances);
            }, 0);
        };

        for (let feature = 0; feature < data[0].length; feature++) {
            const thresholds = Array.from(new Set(data.map(row => row[feature])));
            for (let threshold of thresholds) {
                const groups = this.splitData(data, labels, feature, threshold);
                const giniScore = gini(groups.map(group => group.labels), Array.from(new Set(labels)));
                if (giniScore < bestGini) {
                    bestGini = giniScore;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        return { feature: bestFeature, threshold: bestThreshold };
    }

    splitData(data, labels, feature, threshold) {
        const leftData = [];
        const leftLabels = [];
        const rightData = [];
        const rightLabels = [];

        data.forEach((row, index) => {
            if (row[feature] <= threshold) {
                leftData.push(row);
                leftLabels.push(labels[index]);
            } else {
                rightData.push(row);
                rightLabels.push(labels[index]);
            }
        });

        return [
            { data: leftData, labels: leftLabels },
            { data: rightData, labels: rightLabels },
        ];
    }

    subset(array, indices) {
        return array.filter((_, index) => indices[index]);
    }

    majorityVote(labels) {
        const counts = {};
        labels.forEach(label => counts[label] = (counts[label] || 0) + 1);
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }

    predict(X) {
        const predictRow = (node, row) => {
            if (typeof node === 'number') return node;
            return row[node.feature] <= node.threshold ? predictRow(node.left, row) : predictRow(node.right, row);
        };

        return X.map(row => predictRow(this.tree, row));
    }
}
