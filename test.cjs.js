import { GPU } from 'gpu.js';

// Create a function to generate a 32768x32768 matrix with random values

const generateMatrices = () => {
    const matrices = [[], []]
    for (let y = 0; y < 512; y++){
        matrices[0].push([])
        matrices[1].push([])
        for (let x = 0; x < 512; x++){
            matrices[0][y].push(Math.random())
            matrices[1][y].push(Math.random())
        }
    }
    return matrices
}


const randMat = () => {
    const matrices = []
    for (let y = 0; y < 512; y++) {
        matrices.push(Math.random())
    }
    return matrices
}




console.time('gpu')
const gpu = new GPU({mode:'gpu'});
console.log(gpu.mode)
const multiplyMatrix = gpu.createKernel(function(a, b) {
    let sum = 0;
    for (let i = 0; i < 512; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
}).setOutput([512, 512])


const sort = gpu.createKernel(function(a) {
    const b= a.sort();
    return b;
}).setOutput([512])


const matrices = generateMatrices()
const out = multiplyMatrix(matrices[0], matrices[1])
//ole.log(out[10][12]) // Logs the element at the 10th row and the 12th column of the output matrix

console.timeEnd('gpu');