# YOLOv8 Object Detection and Tracking

The YOLOv8s Stock Market Pattern Detection model is an advanced object detection system built on the YOLO framework, specifically designed to analyze real-time stock market trading videos. By automatically identifying and classifying various chart patterns as they form, this model provides traders and investors with timely insights to support informed decision-making. Fine-tuned on a diverse dataset, it achieves high accuracy in detecting stock market patterns during live trading scenarios. This innovative tool streamlines the chart analysis process, offering valuable assistance to financial professionals by automating the recognition of key market indicators in real-time video data.

![](https://github.com/foduucom/Stockmarket-pattern-detection/blob/main/demo_prediction-result.mp4)

## Overview

Our YOLOv8s-based model revolutionizes stock market analysis by offering real-time pattern detection in live trading video feeds. This cutting-edge tool empowers traders and investors with immediate insights, enhancing decision-making in fast-paced markets.

## Features

- Custom-trained YOLOv8 model for object detection
- Real-time object tracking using ByteTrack
- Video processing with bounding box and label annotations
- Configurable class selection for specific object detection

## Detected Patterns

Head and shoulders bottom
Head and shoulders top
M_Head
StockLine
Triangle
W_Bottom

## How to Get Started with the Model
To begin using the YOLOv8s Stock Market Pattern Detection model on live trading video data, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install the required dependencies:

```bash
pip install ultralytics supervision numpy ipython
```

## Usage

1. Update the `SOURCE_VIDEO_PATH` and `TARGET_VIDEO_PATH` variables in the script with your input and output video paths.

2. Run the code:

```bash
from ultralytics import YOLO
import numpy as np
from IPython import display
import supervision as sv
import ultralytics

SOURCE_VIDEO_PATH = "path/to/your/source/video.mp4"
TARGET_VIDEO_PATH = "path/to/your/output_video.mp4"

display.clear_output()
ultralytics.checks()

# Laoding pre-trained model
model=YOLO('best.pt')

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids 
selected_classes = [0,1,2,3,4,5]
sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create BYTETracker instance
tracker = sv.ByteTrack()

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
   
    # tracking detections
    detections = tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame=box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    return annotated_frame

# process the whole video
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)
print("done")
```

or you can run the script:

```bash
python inference.py
```

4. The processed video with object detection and tracking annotations will be saved to the specified `TARGET_VIDEO_PATH`.

5. ## Customization

- If you load you model to detect specific classes, modify the `selected_classes` list in the script.
- Adjust the annotation styles by modifying the `BoundingBoxAnnotator` and `LabelAnnotator` parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Supervision](https://github.com/roboflow/supervision) for video processing and annotation tools
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for object tracking

