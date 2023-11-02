# YOLO Detector and SiamRPN++ tracker

## Detector
 - `YOLOv5s`
 - `YOLOv8s`

## Tracker
 - `SiamRPN++` tracker

## Testing

Add `is_cuda` argument for running on CUDA.
To test `SiamRPN++`, it needs to unzip `search_net.onnx.zip` and `target_net.onnx.zip`.

`ROI` tracking

```bash
python main_YOLO.py
```

SOT by clicking an object

```bash
python main.py
```