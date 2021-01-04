import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
// import {getData} from '../xor/data'
import {getData} from './data'

// 用简单模型 拟合 复杂数据 就会出现 欠拟合，训练时间不够也会导致欠拟合

// 过拟合应对方法：早停法，权重衰减，丢弃法

window.onload = async () => {
    const data = getData(200, 3)  // 加载 xor数据

    tfvis.render.scatterplot(
        {name: '过拟合欠拟合训练数据'},
        {
            values: [
                data.filter(point => point.label === 1),
                data.filter(point => point.label === 0)
            ]
        }
    )

    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 10,
        activation: 'sigmoid',
        inputShape: [2],
        // kernelRegularizer: tf.regularizers.l2({l2: 1})  // l2 正则化
    }))

    model.add(tf.layers.dropout({rate: 0.9})) // 随机dropout一些unit 解决过拟合

    // 增加一层模型，模拟过拟合情况
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
    }))

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    })

    const input = tf.tensor(data.map(p => [p.x, p.y]))
    const label = tf.tensor(data.map(p => p.label))

    await model.fit(input, label, {
        validationSplit: 0.2,
        epochs: 200,

        callbacks: tfvis.show.fitCallbacks(
            {name: '训练效果'},
            ['loss', 'val_loss'],
            {callbacks: ['onEpochEnd']}
        )
    })

}