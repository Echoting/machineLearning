const IMAGE_SIZE = 224;

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
    for (let i = 0; i < 30; i += 1) {
        ['android', 'apple', 'windows'].forEach(label => {
            const src = `http://127.0.0.1:8080/brand/train/${label}-${i}.jpg`;
            const img = loadImg(src);
            loadImgs.push(img);
            labels.push([
                label === 'android' ? 1 : 0,
                label === 'apple' ? 1 : 0,
                label === 'windows' ? 1 : 0,
            ]);
        });
    }
    const inputs = await Promise.all(loadImgs);
    return {
        inputs,
        labels,
    };
}