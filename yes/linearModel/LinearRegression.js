export default class LinearRegression {
    constructor() {
        this.weights = null;
    }

    fit(X, y) {
        const Xt = Math.transpose(X);
        const XtX = Math.multiply(Xt, X);
        const XtX_inv = Math.inv(XtX);
        const XtX_inv_Xt = Math.multiply(XtX_inv, Xt);
        this.weights = Math.multiply(XtX_inv_Xt, y);
    }

    predict(X) {
        return Math.multiply(X, this.weights);
    }
}
