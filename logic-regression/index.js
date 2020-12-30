import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import {getData} from './data'

window.onload = async () => {
    const data = getData(400)

    tfvis.render.scatterplot(
        {name: '逻辑回归训练数据'},
        {
            values: [
                data.filter(point => point.label === 1),
                data.filter(point => point.label === 0)
            ]
        }
    )

    // 初始化一个连续的模型
    const model = tf.sequential()

    // 给model添加 全连接 层
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid' // 使用sigmoid激活函数 将输出值压缩到 0-1 之间
    }))

    // 使用logLoss计算损失，使用adam优化器
    model.compile({loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

    // 将输入输出转换为tensor，有两个特征的装换方法：tf.tensor([[1, 2], [3, 4]]), 一个特征的转换方法：tf.tensor([1, 2, 3, 4])
    const inputs = tf.tensor(data.map(point => [point.x, point.y]))
    const labels = tf.tensor(data.map(point => point.label))

    await model.fit(inputs, labels, {
        batchSize: 40,  // 每次要学习的样本数量，400个样本，每次学习40个，分10次学习
        epochs: 20, // 迭代整个数据样本的次数，400个样本，训练20次
        callbacks: tfvis.show.fitCallbacks(
            {name: '逻辑回归训练过程'},
            ['loss']
        )
    })

    // 使用 model.predict 进行预测，如果输入为tensor，输出也为tensor
    // const output = model.predict(tf.tensor([[1,2]]))
    // console.log(output.dataSync())

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
        alert(`预测结果为：${pred.dataSync()}`)
    }
}