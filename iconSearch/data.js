const IMAGE_SIZE = 224
const SIZE_NUM = 3
const PLACEMENT_NUM = 9

export const CLASSES_LABEL = [
    'SvgCalendar',
    'SvgChat',
    'SvgDelete',
    'SvgHeart',
    'SvgHeartFill',
    'SvgHome',
    'SvgMagnifierMinus',
    'SvgMagnifierPlus',
    'SvgUserPlus',
    'SvgUserTeam'
]

const loadImg = (src) => {
    return new Promise(resolve => {
        // 等价于 document.createElement('img')
        const img = new Image();
        // 将图片的 crossOrigin 属性设置为"匿名"（即，允许对未经过验证的图像进行跨域下载）
        img.crossOrigin = "anonymous";
        img.src = src;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => resolve(img);
    });
};

export const getInputs = async () => {
    const loadImgs = [];
    const labels = [];

    for (let size = 0; size < SIZE_NUM; size++) {
        for (let place = 0; place < PLACEMENT_NUM; place++) {
            CLASSES_LABEL.forEach(label => {
                const imageSrc = `http://127.0.0.1:8080/icon/train/${label}_${size}_${place}.jpg`
                const img = loadImg(imageSrc)
                loadImgs.push(img)

                const labelArr = CLASSES_LABEL.map(item => {
                    return label === item ? 1 : 0
                })
                labels.push(labelArr);

            })
        }
    }

    const inputs = await Promise.all(loadImgs);
    return {
        inputs,
        labels,
    };
}