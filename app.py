import os
import cv2
import yaml
import torch
import shutil
import numpy as np
import streamlit as st
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap
from torch.utils.data import DataLoader

import clip  # ActionCLIP version
from modules.Visual_Prompt import visual_prompt
from datasets import Action_DATASETS
from utils.Augmentation import get_augmentation
from utils.another_Text_Prompt import text_prompt

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# === Paths ===
CONFIG_PATH = "./configs/config.yaml"
TMP_VAL_LIST = "tmp_val_list.txt"
TMP_LABEL_LIST = "tmp_label_list.csv"
IMAGE_TMPL = "img_{:05d}.jpg"
FRAME_OUTPUT_DIR = "inference_frames"
UPLOADED_VIDEO_DIR = "uploaded_video"
DUMMY_LABEL = 0

@st.cache_resource(show_spinner=False)
def load_model_and_config():
    with open(CONFIG_PATH, 'r') as f:
        config = DotMap(yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout
    )
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    fusion_model = torch.nn.DataParallel(fusion_model).to(device)

    if os.path.isfile(config.pretrain):
        checkpoint = torch.load(config.pretrain, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
    else:
        st.error(f"No checkpoint found at {config.pretrain}")
        return None, None, None

    return config, model.eval(), fusion_model.eval()

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, IMAGE_TMPL.format(frame_count + 1))
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    return frame_count

def run_inference_on_frames(config, model, fusion_model, frame_dir, total_frames, class_labels):
    device = next(model.parameters()).device

    with open(TMP_VAL_LIST, 'w') as f:
        f.write(f"{frame_dir} {total_frames} {DUMMY_LABEL}\n")
    with open(TMP_LABEL_LIST, 'w') as f:
        f.write("\n".join(class_labels))

    transform_val = get_augmentation(False, config)
    val_data = Action_DATASETS(
        list_file=TMP_VAL_LIST,
        labels_file=TMP_LABEL_LIST,
        num_segments=config.data.num_segments,
        image_tmpl=IMAGE_TMPL,
        transform=transform_val,
        random_shift=False
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    classes, num_text_aug, _ = text_prompt(val_data)
    text_features = model.encode_text(classes.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        for image_batch, _ in tqdm(val_loader, desc="Inferring"):
            image_batch = image_batch.to(device)
            b, tc, h, w = image_batch.shape
            t = config.data.num_segments
            c = 3
            image_input = image_batch.view(b, t, c, h, w).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            probs, indices = similarity.topk(1, dim=-1)

            predicted_class = val_data.classes[indices.item()][0]
            confidence = probs.item() * 100
            return predicted_class, confidence

# === Streamlit UI ===
st.set_page_config(page_title="ActionCLIP UI", layout="centered")
st.title("🎥 ActionCLIP Inference UI")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
class_input = st.text_input("Enter class labels (comma-separated)", "Shooting,Assault,Explosion")

# Load model only once
config, model, fusion_model = load_model_and_config()

if uploaded_video and class_input:
    Path(UPLOADED_VIDEO_DIR).mkdir(exist_ok=True)
    video_path = os.path.join(UPLOADED_VIDEO_DIR, uploaded_video.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    if st.button("🔍 Run Inference"):
        with st.spinner("Extracting frames..."):
            shutil.rmtree(FRAME_OUTPUT_DIR, ignore_errors=True)
            total_frames = extract_frames(video_path, FRAME_OUTPUT_DIR)

        with st.spinner("Running inference..."):
            class_list = [c.strip() for c in class_input.split(",")]
            prediction, conf = run_inference_on_frames(config, model, fusion_model, FRAME_OUTPUT_DIR, total_frames, class_list)

        if prediction:
            st.success(f"✅ Predicted Class: **{prediction}** ({conf:.2f}%)")
        else:
            st.error("❌ Inference failed.")

        # Optional cleanup
        shutil.rmtree(FRAME_OUTPUT_DIR, ignore_errors=True)
        os.remove(video_path)
        shutil.rmtree(UPLOADED_VIDEO_DIR, ignore_errors=True)
