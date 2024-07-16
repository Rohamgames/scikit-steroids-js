function accuracyScore(y_true, y_pred) {
    if (y_true.length !== y_pred.length) {
        throw new Error('True and predicted labels must have the same length.');
    }

    let correct = 0;
    for (let i = 0; i < y_true.length; i++) {
        if (y_true[i] === y_pred[i]) {
            correct++;
        }
    }

    const accuracy = correct / y_true.length;

    return accuracy * 100;
}