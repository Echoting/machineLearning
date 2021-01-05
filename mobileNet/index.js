import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import {file2img} from './utils'

import {IMAGENET_CLASSES} from './imagenet_classes'

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json'

window.onload = async () => {

    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH)

    window.predict = async file => {
        const image = await file2img(file)
        document.body.appendChild(image)
        const pred = tf.tidy(() => {
            const input = tf.browser.fromPixels(image)
                .toFloat()
                .sub(255/2)
                .div(255/2) // 归一化 到 -1 ~ 1之间
                .reshape([1, 224, 224, 3]) // 是个彩色图片

            return model.predict(input)
        })

        console.log(pred)

        const index = pred.argMax(1).dataSync()[0]

        setTimeout(() => {
            alert('预测结果为: ' + IMAGENET_CLASSES[index])
        }, 0)
    }

}