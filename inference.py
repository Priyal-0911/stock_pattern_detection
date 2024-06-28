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
selected_classes = [0, 1,2,3,4,5]
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