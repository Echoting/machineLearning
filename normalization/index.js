import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

window.onload = async function () {
    const heights = [160, 165, 170, 175, 180]
    const weights = [40, 50, 60, 70, 80]

    tfvis.render.scatterplot(
        {name: '身高体重训练数据'},
        {values: heights.map((height, index) => ({x: height, y: weights[index]}))},
        {xAxisDomain: [140, 200], yAxisDomain: [30, 90]}
    )

    // 3.1 归一化
    // Min-Max Normalization
    // x' = (x - X_min) / (X_max - X_min)
    const input = tf.tensor(heights).sub(150).div(30)
    const labels = tf.tensor(weights).sub(40).div(40)
    input.print()
    labels.print()

    // 初始化一个连续的模型
    const model = tf.sequential()

    // 给model添加 全连接 层
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))

    // 使用均方误差损失函数  使用随机梯度下降优化器
    model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1)})

    await model.fit(input, labels, {
        batchSize: 5,
        epochs: 400,
        callbacks: tfvis.show.fitCallbacks(
            {name: '身高体重训练过程'},
            ['loss']
        )
    })

    // 使用 model.predict 进行预测，如果输入为tensor，输出也为tensor
    // 输入的数据也要做归一化
    const output = model.predict(tf.tensor([185]).sub(150).div(30)) // 预测 x = 5的值
    // output.print()

    // 使用dataSync将tensor数据转换为普通数据
    // 输出的数据做反归一化
    console.log(output.mul(40).add(40).dataSync())
}