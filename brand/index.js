import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

import {getInputs} from './data'
import {img2x, file2img} from './utils'

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json'
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
    const {inputs, labels} = await getInputs()

    // 使用surface 展示图片
    const surface = tfvis.visor().surface({name: 'My Surface', styles: {height: 300}})
    inputs.forEach(item => {
        surface.drawArea.appendChild(item)
    })

    // 加载imageNet模型
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH)
    // mobilenet.summary()
    // 截断 imageNet模型，生成我们新的 截断模型 从 conv_pw_13_relu 截断
    const layer = mobilenet.getLayer('conv_pw_13_relu')  // 获取卷积层 名称为 conv_pw_13_relu 的层
    const truncatMobileNet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    })

    // 生成了一个新的模型，一个flatten层 两个dense层
    const model = tf.sequential()
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)
    }))
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));
    // 设置损失函数和优化器
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam()
    })

    // 将截断模型和我们的新模型结合起来训练 ---------

    // 将图片喂给截断模型truncatMobileNet进行predict，拿到输出的imageXs
    const {imageXs, labelYs} = tf.tidy(() => {
        // tf.concat将多个tensor组成的数组合并成一个tensor
        const imageXs = tf.concat(inputs.map(imgEl => {
            return truncatMobileNet.predict(img2x(imgEl))
        }))

        const labelYs = tf.tensor(labels)

        return {
            imageXs,
            labelYs
        }
    })

    // 将经过截断模型的 imageXs 和 labelYs拿给新模型model进行训练
    await model.fit(imageXs, labelYs, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    })

    window.predict = async file => {
        // 将上传的图片展示到页面上
        const image = await file2img(file)
        document.body.appendChild(image)
        const pred = tf.tidy(() => {
            // 将上传的image图片转换成tensor并reshape
            const x = img2x(image)
            // 将图片喂给截断模型进行预测
            const input = truncatMobileNet.predict(x)
            // 将截断模型输出的结果喂给新模型预测  这个预测过程与训练过程保持一致
            return model.predict(input)
        })

        const index = pred.argMax(1).dataSync()[0]

        setTimeout(() => {
            alert('预测结果为: ' + BRAND_CLASSES[index])
        }, 0)
    }



}