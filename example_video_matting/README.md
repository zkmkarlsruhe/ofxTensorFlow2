# Video Matting Example
Realtime background substraction for video or webcam using MobileNetV3. 
Model provided by https://github.com/PeterL1n/RobustVideoMatting

Shout out to [Natxopedreira 2021](https://github.com/natxopedreira) for providing the initial example.

![](../media/video_matting.gif)

### Notes
Please check out the [official documentation](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md) for this model.

Starting with the Turing architecture (e.g. RTX 2070), GPUs can utilize the FP16 data type and are therefore much faster!

##### Choosing the downsample factor
The table provides a general guideline. Please adjust based on your video content.

| Resolution    | Portrait      | Full-Body      |
| ------------- | ------------- | -------------- |
| <= 512x512    | 1             | 1              |
| 1280x720      | 0.375         | 0.6            |
| 1920x1080     | 0.25          | 0.4            |
| 3840x2160     | 0.125         | 0.2            |

Internally, the model resizes down the input for stage 1. Then, it refines at high-resolution for stage 2.

Set `downsample_ratio` so that the downsampled resolution is between 256 and 512. For example, for `1920x1080` input with `downsample_ratio=0.25`, the resized resolution `480x270` is between 256 and 512.



### Other References

[Video sample](   
https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU)

[Background image](  
https://www.pexels.com/es-es/foto/capsulas-blancas-sobre-fondo-amarillo-3683056/)