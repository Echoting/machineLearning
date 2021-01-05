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

}