import nj from 'numjs';

// Function to split data and labels into training and testing sets
export default function trainTestSplit(data, labels, test_size = 0.2, random_state = null) {
    // Convert data and labels to NumJS arrays if they are not already
    data = nj.array(data);
    labels = nj.array(labels);

    // Determine number of samples
    const numSamples = data.shape[0];

    // Calculate the number of samples for training and testing
    const numTrain = Math.floor(numSamples * (1 - test_size));
    const numTest = numSamples - numTrain;

    // Create shuffled indices
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    shuffleArray(indices);

    // Split data and labels
    const X_train = nj.zeros([numTrain, ...data.shape.slice(1)]);
    const y_train = nj.zeros([numTrain]);
    const X_test = nj.zeros([numTest, ...data.shape.slice(1)]);
    const y_test = nj.zeros([numTest]);

    for (let i = 0; i < numSamples; i++) {
        const sampleIndex = indices[i];
        const sampleData = data.pick(sampleIndex, null);
        const sampleLabel = labels.get(sampleIndex);

        if (i < numTrain) {
            X_train.set(i, sampleData);
            y_train.set(i, sampleLabel);
        } else {
            X_test.set(i - numTrain, sampleData);
            y_test.set(i - numTrain, sampleLabel);
        }
    }

    return { X_train, X_test, y_train, y_test };
}

// Fisher-Yates shuffle implementation
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
