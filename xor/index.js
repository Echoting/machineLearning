import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import {getData} from './data'

window.onload = async () => {
    const data = getData(400)

    tfvis.render.scatterplot(
        {name: 'xor训练数据'},
        {
            values: [
                data.filter(point => point.label === 1),
                data.filter(point => point.label === 0)
            ]
        }
    )

    const model = tf.sequential()
    // 增加第一层，隐藏层，设置
    model.add(tf.layers.dense({
        units: 4,  // 设置神经元个数
        inputShape: [2],  // 第一层需要设置 inputShape
        activation: 'relu'
    }))

    // 增加输出层
    model.add(tf.layers.dense({
        units: 1, // 神经元个数为1 上一层的输出为这一层的输入，所以inputShape不用设置，这里会自动计算
        activation: 'sigmoid' // 要输出一个0-1之间的概率，所以只能使用sigmoid激活函数
    }))

    // 使用logLoss计算损失，使用adam优化器
    model.compile({loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

    const inputs = tf.tensor(data.map(point => [point.x, point.y]))
    const labels = tf.tensor(data.map(point => point.label))

    await model.fit(inputs, labels, {
        // batchSize: 40,
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'xor训练过程'},
            ['loss']
        )
    })

    window.predict = async (form) => {
        const pred = await model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
        alert(`预测结果为：${pred.dataSync()}`)
    }


}