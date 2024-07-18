import * as math from 'mathjs'
import {GPU} from 'gpu.js'
export default class LinearRegression {
    constructor() {
        this.weights = null;
        this.gpu = new GPU();
    }

    fit(X, y) {
        console.time('transpose');
        const Xt = math.transpose(X);
        console.timeEnd('transpose');
        console.time('multiply');
        const XtX = math.multiply(Xt, X);
        console.timeEnd('multiply');
        console.time('inv');
        const XtX_inv = math.inv(XtX);
        console.timeEnd('inv');
        console.time('m2');
        const XtX_inv_Xt = math.multiply(XtX_inv, Xt);
        console.timeEnd('m2');
        console.time('we');
        this.weights = math.multiply(XtX_inv_Xt, y);
        console.timeEnd('we');
    }

    predict(X) {
        return math.multiply(X, this.weights);
    }
}
