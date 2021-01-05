import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

import {MnistData} from './data'

window.onload = async () => {
    const data = new MnistData()
    await data.load()
    const examples = data.nextTestBatch(20)  // example 数据格式是tensor

    console.log(examples)

    // 使用tfvis visor展示图片
    const surface = tfvis.visor().surface({name: 'My Surface', tab: 'My Tab'});

    // 将tensor数据转换为图片
    for (let i = 0; i < 20; i++) {
        // 从数据集中分割出每张图片
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, 784])
                .reshape([28, 28, 1])
        })

        const canvas = document.createElement('canvas')
        canvas.width = 28
        canvas.height = 28
        canvas.style = 'margin: 4px'
        // 将图片转换成像素
        await tf.browser.toPixels(imageTensor, canvas)

        // 在document的body中展示图片或者使用tfvis visor展示图片
        // document.body.appendChild(canvas)
        surface.drawArea.appendChild(canvas)
    }

    const model = tf.sequential()
    // 增加卷积层
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))
    // 最大池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }))

    // 卷积层
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))
    // 最大池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }))

    // 将高维数据摊平
    model.add(tf.layers.flatten())

    // 全连接层
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }))


    // 设置损失函数和优化器
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['accuracy']
    })

    // 训练集
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(1000)
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ]
    })

    // 验证集
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200)
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ]
    })

    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        batchSize: 500,
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    })

    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX, e.offsetY, 15, 15);
        }
    });

    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);
    };

    clear();


    window.predict = () => {
        const input = tf.tidy(() => {
            // 图片是300*300的，需要将图片resize 成 28*28
            return tf.image.resizeBilinear(
                // 将图片转换为tensor
                tf.browser.fromPixels(canvas),
                [28, 28],
                true
            ).slice([0, 0, 0], [28, 28, 1])  // 图片看似是黑白的，但是还是要灰度处理一下，彩色图片是3个通道，reshape到1个通道
                .toFloat()
                .div(255) // 数据进行归一化
                .reshape([1, 28, 28, 1]);  // 数据还要reshape一下
        })

        const pred = model.predict(input).argMax(1);
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    }

}