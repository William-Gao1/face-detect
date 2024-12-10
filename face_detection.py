import cv2 as cv
from skimage.metrics import structural_similarity
from tqdm import tqdm
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import os
from os.path import dirname, abspath
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-video', help="File path of input video", required=True)
parser.add_argument('--target-face', help="File path of target face image", required=True)
parser.add_argument('--output-dir', help="Directory to put output files", required=True)

args = parser.parse_args()

d = dirname(abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

processing_width = 854
processing_height = 480

detector = cv.FaceDetectorYN.create(os.path.join(d, "face_detection_yunet_2023mar.onnx"), config="", input_size=(processing_width, processing_height))
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0)

def get_face_embedding(image):
    preprocessed = torch.from_numpy(cv.resize(image, (160, 160))).moveaxis(2, 0).float().to(device)
    preprocessed = (preprocessed - 127.5) / 128
    return resnet(preprocessed.unsqueeze(0))

def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)

def did_scene_change(prev_frame, cur_frame, threshold=0.7):
    grey_img = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)
    resized_img_for_ssim = cv.resize(grey_img, (192, 144), interpolation=cv.INTER_AREA)

    if prev_frame is None:
        _ = structural_similarity(resized_img_for_ssim, resized_img_for_ssim)
        return resized_img_for_ssim, False

    ssim = structural_similarity(resized_img_for_ssim, prev_frame)
    return resized_img_for_ssim, ssim < threshold

def get_target_face(target_face_emb, img, face_bounding_boxes):
    target_face = None
    target_face_idx = -1
    similarities = []
    for idx, face in enumerate(face_bounding_boxes):
        x, y, width, height = face[:4].astype(int)
        cropped_face = img[y:y+height, x:x+width, :]

        potential_embedding = get_face_embedding(cropped_face)
        emb_similarity = cosine_similarity(target_face_emb, potential_embedding)
        similarities.append(emb_similarity.item())
        if emb_similarity > 0.5 and (target_face is None or target_face["similarity"] < emb_similarity):
            target_face = {
                 "x": int(x),
                 "y": int(y),
                 "width": int(width),
                 "height": int(height),
                 "similarity": float(emb_similarity.item())
            }
            target_face_idx = idx
    return target_face, target_face_idx, similarities

def scale_bounding_boxes(face_bounding_boxes, w_ratio, h_ratio):
    scaled_boxes = []
    for face in face_bounding_boxes:
        x, y, width, height = face[:4].astype(int)
        x = int(x * w_ratio)
        y = int(y * h_ratio)
        width = int(width * w_ratio)
        height = int(height * h_ratio)
        scaled_face = np.array([x, y, width, height, face[-1]], dtype=face.dtype)
        scaled_boxes.append(scaled_face)
    return np.array(scaled_boxes)

def draw_face_bounding_boxes(img, face_bounding_boxes, target_face_idx, similarities):
    img = img.copy()
    for idx, (face, similarity) in enumerate(zip(face_bounding_boxes, similarities)):
        x, y, width, height = face[:4].astype(int)

        colour = (0, 255, 0) if idx == target_face_idx else (0, 0, 255)
        cv.rectangle(img, (x, y), (x + width, y + height), colour, 2)
        cv.putText(img, f'{similarity:.2f}', (x, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour)

        if idx == target_face_idx:
            cv.putText(img, "TARGET", (x, y+30), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour)
    return img

face = cv.imread(args.target_face)

resized_face = cv.resize(face, (processing_width, processing_height), interpolation=cv.INTER_AREA)
_, faces = detector.detect(resized_face)
coords = faces[0][:-1].astype(np.int32)
face_emb = get_face_embedding(resized_face[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2], :])

cap = cv.VideoCapture(args.input_video)
fps = cap.get(cv.CAP_PROP_FPS)
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Scene management
scene = 0
scene_metadata = []
scene_start_frame = 0

# Keep track of all scenes in one place
all_scenes = []

output_video_path = os.path.join(args.output_dir, 'annotated_video.mp4')
video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

prev_frame = None

# Variables for target face scene video
target_face_video_writer = None
target_face_video_path = None
target_face_dimensions = None
target_face_detected_this_scene = False

def finalize_scene(scene_num, scene_metadata, start_frame, end_frame):
    if len(scene_metadata) == 0:
        return None
    start_time = start_frame / fps
    end_time = end_frame / fps
    scene_info = {
        "scene_number": scene_num,
        "file_name": os.path.basename(output_video_path),
        "start_time": start_time,
        "end_time": end_time,
        "frames": scene_metadata
    }
    return scene_info

for i in tqdm(range(num_frames)):
    ret, img = cap.read()
    if not ret:
        break

    resized_img = cv.resize(img, (processing_width, processing_height), interpolation=cv.INTER_AREA)
    prev_frame, scene_change = did_scene_change(prev_frame, resized_img, threshold=0.7)

    if scene_change and i > 0:
        # Close current scene
        scene_info = finalize_scene(scene, scene_metadata, scene_start_frame, i-1)
        if scene_info is not None:
            all_scenes.append(scene_info)

        # Close target face video if it was opened
        if target_face_video_writer is not None:
            target_face_video_writer.release()
            target_face_video_writer = None
            target_face_dimensions = None
            target_face_detected_this_scene = False

        # Start a new scene
        scene += 1
        scene_metadata = []
        scene_start_frame = i

    # Detect faces
    _, detected_faces = detector.detect(resized_img)

    target_face = None
    target_face_idx = -1
    similarities = []
    if detected_faces is not None and len(detected_faces) > 0:
        target_face, target_face_idx, similarities = get_target_face(face_emb, resized_img, detected_faces)
        scaled_faces = scale_bounding_boxes(detected_faces, width/processing_width, height/processing_height)
        annotated_image = draw_face_bounding_boxes(img, scaled_faces, target_face_idx, similarities)
    else:
        annotated_image = img

    # Write frame to main scene video
    video_writer.write(annotated_image)

    # Handle target face video
    if target_face is not None:
        # Scale target_face coordinates to original image size
        x_scaled = int(target_face["x"] * (width / processing_width))
        y_scaled = int(target_face["y"] * (height / processing_height))
        w_scaled = int(target_face["width"] * (width / processing_width))
        h_scaled = int(target_face["height"] * (height / processing_height))

        # Update target_face dict with scaled coordinates
        target_face["x"] = x_scaled
        target_face["y"] = y_scaled
        target_face["width"] = w_scaled
        target_face["height"] = h_scaled

        # If we haven't started a target face video for this scene yet, do it now
        if not target_face_detected_this_scene:
            tw, th = w_scaled, h_scaled
            tw = max(tw, 10)
            th = max(th, 10)
            target_face_dimensions = (tw, th)
            target_face_video_path = os.path.join(args.output_dir, f'target_face_scene_{scene}.mp4')
            target_face_video_writer = cv.VideoWriter(
                target_face_video_path,
                cv.VideoWriter_fourcc(*'mp4v'),
                fps,
                (tw, th)
            )
            target_face_detected_this_scene = True

        # Write the target face frame to the target video
        face_cropped = img[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
        if (w_scaled, h_scaled) != target_face_dimensions:
            face_cropped = cv.resize(face_cropped, target_face_dimensions)
        target_face_video_writer.write(face_cropped)
    # Add per-frame metadata
    frame_metadata = {
        "frame_number": i,
        "target_face": target_face if target_face is not None else None
    }
    scene_metadata.append(frame_metadata)

# End of video
cap.release()
video_writer.release()

# Save the last scene's metadata
scene_info = finalize_scene(scene, scene_metadata, scene_start_frame, num_frames-1)
if scene_info is not None:
    all_scenes.append(scene_info)

if target_face_video_writer is not None:
    target_face_video_writer.release()

# Write all scenes metadata to a single JSON
output_json_path = os.path.join(args.output_dir, 'metadata.json')
with open(output_json_path, 'w') as f:
    json.dump({"scenes": all_scenes}, f, indent=4)
