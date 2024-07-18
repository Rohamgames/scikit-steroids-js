import * as math from 'mathjs'
import {GPU} from 'gpu.js'

export default class LinearRegression {
    constructor(mode='cpu') {
        this.weights = null;
        this.gpu = new GPU({ mode });
        // TODO: implement GPU
    }

    fit(X, y) {
        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const XtX_inv_Xt = math.multiply(XtX_inv, Xt);
        this.weights = math.multiply(XtX_inv_Xt, y);
    }

    predict(X) {
        return math.multiply(X, this.weights);
    }
}
