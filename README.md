# Face Detection and Scene Processing Pipeline

This project is a Python-based pipeline designed for face detection, face recognition, and scene segmentation in videos. The goal is to identify a target face in a video, annotate the video with bounding boxes and similarities, and segment the video into scenes based on changes in content.

---

## Features

1. **Scene Change Detection**:
   - Detects significant changes between consecutive frames using Structural Similarity Index (SSIM).
   - Automatically segments the video into scenes.

2. **Face Detection**:
   - Uses a pre-trained ONNX-based face detection model (`face_detection_yunet_2023mar.onnx`).

3. **Face Recognition**:
   - Employs the `MTCNN` for face alignment and the `InceptionResnetV1` model for face embeddings.
   - Matches detected faces with a target face using cosine similarity.

4. **Video Annotation**:
   - Annotates video frames with bounding boxes and similarity scores.
   - Highlights the target face distinctly.

5. **Scene-wise Target Face Cropping**:
   - Extracts and saves the target face as separate videos for each scene where it is detected.

6. **Metadata Generation**:
   - Outputs a JSON file containing metadata for all scenes, including timestamps, detected faces, and the presence of the target face.

---

## Dependencies

The pipeline is implemented with the following Python packages:

- **`cv2` (OpenCV)**: Used for video processing, image resizing, and visualization.
- **`skimage`**: For computing SSIM to detect scene changes.
- **`tqdm`**: For progress bar visualization.
- **`numpy`**: For array manipulation.
- **`torch`**: For running face embedding models.
- **`facenet-pytorch`**: Provides pre-trained models (`MTCNN` and `InceptionResnetV1`).

Installation of dependencies can be done by

```bash
pip3 install -r requirements.txt
```

> **Note**: Pre-trained models and external libraries are only used for face detection and embeddings. The pipeline logic, including scene detection, annotation, and processing, is hand-written.

---

## Usage

Run the script using the command line with the required arguments:

```bash
python main.py --input-video <path_to_video> --target-face <path_to_face_image> --output-dir <output_directory>
```

### Arguments

- `--input-video`: Path to the input video file.
- `--target-face`: Path to the target face image file.
- `--output-dir`: Directory where the output files (annotated video, cropped videos, metadata) will be saved.

---

## Output

1. **Annotated Video**:
   - A video file with bounding boxes around detected faces.
   - The target face is labeled as "TARGET" with its similarity score.

2. **Target Face Videos**:
   - Cropped videos for each scene where the target face is detected.

3. **Metadata**:
   - A JSON file (`metadata.json`) containing:
     - Scene numbers.
     - Timestamps.
     - Frame-wise details, including detected faces and similarity scores.

---

## Key Functions

1. **Face Detection and Recognition**:
   - `get_face_embedding(image)`: Extracts embeddings for a face using `InceptionResnetV1`.
   - `cosine_similarity(emb1, emb2)`: Computes similarity between two embeddings.
   - `get_target_face(target_face_emb, img, face_bounding_boxes)`: Identifies the target face in a given frame.

2. **Scene Management**:
   - `did_scene_change(prev_frame, cur_frame, threshold)`: Detects scene changes based on SSIM.
   - `finalize_scene(scene_num, scene_metadata, start_frame, end_frame)`: Finalizes and saves metadata for a scene.

3. **Annotation**:
   - `draw_face_bounding_boxes(img, face_bounding_boxes, target_face_idx, similarities)`: Draws bounding boxes and labels on video frames.

---

## Example

```bash
python main.py \
    --input-video examples/debate/inputs/debate.mp4 \
    --target-face  examples/debate/inputs/trump.jpg \
    --output-dir  examples/debate/outputs
```

After running the script:
- The annotated video will be saved as `examples/debate/outputs/annotated_video.mp4`.
- Target face videos will be named like `examples/debate/outputs/target_face_scene_<scene_number>.mp4`.
- Metadata will be saved as `examples/debate/outputs/metadata.json`.

---

## Limitations

1. **Performance**:
   - The pipeline processes frames sequentially, which makes it slower than real-time video playback.
   - For long videos, the processing time can significantly increase.

2. **Complex Scene Changes**:
   - The scene detection mechanism relies on SSIM, which struggles to handle subtle or complex changes (e.g., gradual lighting shifts or zoom effects).
   - This can lead to either missed scene changes or false positives. For example, in the `mkbhd` example, there is a complex scene change which SSIM struggles to handle. As a result there are a large number of sequential scene changes back-to-back.

---

## Notes

1. This pipeline requires a GPU for optimal performance. Ensure CUDA is available on your system.
2. The `face_detection_yunet_2023mar.onnx` file should be located in the script's directory.
