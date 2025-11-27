# ShakeSensorSuperResolution
相机摇摇乐 外部实现 本仓库的目标是通过随手拍摄的图片的堆叠，将相机分辨率推高至镜头分辨率，而不再受传感器的限制 Camera Shake Fun - External Implementation. The goal of this repository is to push the camera resolution up to the lens resolution by stacking casually captured images, no longer limited by the sensor.

**PS_dng_align_avg.py效果 中心解析度**
![alt text](assert/e686e4817c3420d5b6d422b84d8796ba.png)


**三维采样**

- 修改```3d_raw_near_sample.py```文件开头参数，输入raw文件夹，输出16位色深png地址，放大倍率，是否旋转180度（对有的raw种类）
- 暂不支持曲面拟合采样，目前为最近点
- 暂不支持cuda加速

- 或者：
```
1. raw2ply.py
2. plyRGB23.py
3. ply2jpgfast.py
```





