export function file2img(file) {
    return new Promise(resolve => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.width = 224;
            img.height = 224;
            img.onload = () => resolve(img);
        };
    });
}