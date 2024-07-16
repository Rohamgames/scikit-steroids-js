function shuffleArray(array, seed) {
    const shuffledArray = [...array];
    let currentIndex = shuffledArray.length, temporaryValue, randomIndex;

    // Use a consistent seed if provided
    if (seed !== undefined) {
        const rng = new Math.seedrandom(seed.toString());
        while (0 !== currentIndex) {
            randomIndex = Math.floor(rng() * currentIndex);
            currentIndex -= 1;
            temporaryValue = shuffledArray[currentIndex];
            shuffledArray[currentIndex] = shuffledArray[randomIndex];
            shuffledArray[randomIndex] = temporaryValue;
        }
    } else {
        // Standard random shuffle
        while (0 !== currentIndex) {
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;
            temporaryValue = shuffledArray[currentIndex];
            shuffledArray[currentIndex] = shuffledArray[randomIndex];
            shuffledArray[randomIndex] = temporaryValue;
        }
    }

    return shuffledArray;
}

// Function to split features and labels into training and testing sets
export default function trainTestSplit(features, labels, testSize, randomState) {
    // Combine features and labels into a single array for shuffling
    const combinedData = features.map((value, index) => ({ feature: value, label: labels[index] }));

    // Shuffle the combined data array using custom shuffle function with random state
    const shuffledData = randomState ? shuffleArray(combinedData, randomState) : shuffleArray(combinedData);

    // Calculate split index
    const splitIndex = Math.floor(shuffledData.length * (1 - testSize));

    // Split the combined data into training and testing sets
    const trainingSet = shuffledData.slice(0, splitIndex);
    const testingSet = shuffledData.slice(splitIndex);

    // Extract features and labels from training and testing sets
    const X_train = trainingSet.map(data => data.feature);
    const y_train = trainingSet.map(data => data.label);
    const X_test = testingSet.map(data => data.feature);
    const y_test = testingSet.map(data => data.label);

    return { X_train, X_test, y_train, y_test };
}