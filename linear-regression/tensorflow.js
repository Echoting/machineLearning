import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

window.onload = async function () {

    const xs = [1, 2, 3, 4]
    const ys = [1, 3, 5, 7]

    tfvis.render.scatterplot(
        {name: '线性回归训练集'},
        {values: xs.map((x, index) => ({x, y: ys[index]}))},
        {xAxisDomain: [0, 5], yAxisDomain: [0, 8]}
    )

    // 初始化一个连续的模型
    const model = tf.sequential()

    // 给model添加 全连接 层
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))

    // 使用均方误差损失函数  使用随机梯度下降优化器
    model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1)})

    const input = tf.tensor(xs)
    const labels = tf.tensor(ys)

    await model.fit(input, labels, {
        batchSize: 4,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    })

    // 使用 model.predict 进行预测，如果输入为tensor，输出也为tensor
    const output = model.predict(tf.tensor([5])) // 预测 x = 5的值
    // output.print()

    // 使用dataSync将tensor数据转换为普通数据
    console.log(output.dataSync())

}