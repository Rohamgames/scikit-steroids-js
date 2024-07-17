export default class LinearRegression {
    constructor() {
        this.weights = null;
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
