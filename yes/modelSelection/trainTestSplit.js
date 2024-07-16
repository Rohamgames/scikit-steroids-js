import nj from 'numjs';

// Function to split data and labels into training and testing sets
export default function trainTestSplit(data, labels, test_size = 0.2, random_state = null) {
    // Convert data and labels to NumJS arrays if they are not already
    data = nj.array(data);
    labels = nj.array(labels);

    // Combine data and labels into one array of objects for shuffling
    let combined = [];
    for (let i = 0; i < data.shape[0]; i++) {
        combined.push({ data: data.pick(i, null), label: labels.get(i) });
    }

    // Shuffle the combined array using Fisher-Yates algorithm
    combined = shuffleArray(combined);

    // Calculate the split index based on test_size
    let splitIndex = Math.floor(combined.length * (1 - test_size));

    // Split into training and testing sets
    let trainingData = combined.slice(0, splitIndex).map(item => item.data);
    let trainingLabels = combined.slice(0, splitIndex).map(item => item.label);
    let testData = combined.slice(splitIndex).map(item => item.data);
    let testLabels = combined.slice(splitIndex).map(item => item.label);

    // Convert to NumJS arrays
    trainingData = nj.array(trainingData);
    trainingLabels = nj.array(trainingLabels);
    testData = nj.array(testData);
    testLabels = nj.array(testLabels);

    return { X_train: trainingData, X_test: testData, y_train: trainingLabels, y_test: testLabels };
}

// Fisher-Yates shuffle implementation
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}