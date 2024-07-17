import { readFile } from 'fs/promises';
import path from 'path';

const isNode = typeof process !== 'undefined' && process.versions != null && process.versions.node != null;

export default async function loadIris() {
    let fileContent;

    if (isNode) {
        // Node.js environment
        const csvFilePath = path.resolve('./yes/datasets/data/iris.data.csv');
        fileContent = await readFile(csvFilePath, { encoding: 'utf-8' });
    } else {
        // Browser environment
        const response = await fetch('./yes/datasets/data/iris.data.csv'); // Adjust the path
        fileContent = await response.text();
    }

    // Parse the CSV content
    const records = parseCSV(fileContent);

    // Extract features and target
    const featureNames = ['sepal length', 'sepal width', 'petal length', 'petal width'];
    const targetNames = ['setosa', 'versicolor', 'virginica'];

    const data = [];
    const target = [];

    records.forEach(record => {
        data.push([
            parseFloat(record['sepal length']),
            parseFloat(record['sepal width']),
            parseFloat(record['petal length']),
            parseFloat(record['petal width'])
        ]);
        target.push(targetNames.indexOf(record['species']));
    });

    return {
        data,
        target,
        featureNames,
        targetNames
    };
}

function parseCSV(content) {
    const rows = content.split('\n').filter(row => row.trim() !== '');
    const headers = rows[0].split(',');
    const records = rows.slice(1).map(row => {
        const values = row.split(',');
        return headers.reduce((obj, header, i) => {
            obj[header.trim()] = values[i].trim();
            return obj;
        }, {});
    });
    console.log(records)
    return records;
}