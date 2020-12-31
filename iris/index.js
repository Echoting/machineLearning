import {getIrisData, IRIS_CLASSES} from './data'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {

    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15)

    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [4],
        activation: 'sigmoid'
    }))

    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }))

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']
    })

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            {name: '鸢尾花训练效果'},
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd']}
        )
    })

    window.predict = form => {
        const input = tf.tensor([
            [
                form.a.value * 1,
                form.b.value * 1,
                form.c.value * 1,
                form.d.value * 1,
            ]
        ])

        const pred = model.predict(input)

        alert(`预测结果为：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`)

    }

}