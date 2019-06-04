# 黑白图像的直方图均衡化
> 2019.05.15

## Compile command

```bash
g++ 1.cpp `pkg-config opencv --libs` -o bw 
```

## Usage

```bash
./histogram $IMG_FILE_NAME # 直方图均衡化(Grayscale)
./imshow $IMG_FILE_NAME # 仅显示原图片
```



