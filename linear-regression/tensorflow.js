import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = function () {

    const xs = [1, 2, 3, 4]
    const ys = [1, 3, 5, 7]

    tfvis.render.scatterplot(
        {name: '线性回归训练集'},
        {values: xs.map((x, index) => ({x, y: ys[index]}))},
        {xAxisDomain: [0, 5], yAxisDomain: [0, 8]}
    )
}