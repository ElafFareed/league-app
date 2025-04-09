# === IMPORTS ===
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import face_recognition
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from skimage.metrics import structural_similarity as ssim

# === SESSION STATE ===
if "emotion_analysis_done" not in st.session_state:
    st.session_state["emotion_analysis_done"] = False

# === SECTION 1: Load & Merge Stats ===
st.title("🏆 MatchMind dashboard – Player Analysis")

@st.cache_data
def load_data():
    df_stats = pd.read_csv("data/player_stats.csv")
    df_misc = pd.read_csv("data/player_misc.csv")
    df_playingtime = pd.read_csv("data/player_playingtime.csv")
    df_merged = df_stats.merge(df_misc, on=["player", "team"], how="outer")
    df_merged = df_merged.merge(df_playingtime, on=["player", "team"], how="outer")
    df_merged.fillna(0, inplace=True)
    return df_merged

player_df = load_data()

# === SECTION 2: Fatigue & Injury Risk ===
#st.subheader("⚙️ Fatigue Score Computation")
features = ["minutes_90s", "tackles", "press", "blocks", "passes_completed", "miscontrols", "dispossessed"]
features = [f for f in features if f in player_df.columns]
scaler = MinMaxScaler()
player_df["fatigue_score"] = scaler.fit_transform(player_df[features]).mean(axis=1)
player_df["injury_risk"] = player_df["fatigue_score"].apply(lambda x: 1 if x > 0.7 else (0.5 if x > 0.5 else 0))

# === SECTION 3: Load Face Recognition + ViT Emotion Model ===
#st.subheader(" Load Face Match & Emotion Model")

def load_known_faces(folder):
    known = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(folder, filename)
            name = os.path.splitext(filename)[0]
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known[name] = encs[0]
    return known

argentina_faces = load_known_faces("argentina")
france_faces = load_known_faces("france")
all_faces = {**argentina_faces, **france_faces}

mtcnn = MTCNN(keep_all=True, device='cpu')
model_name = "trpakov/vit-face-expression"
vit_model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
st.success(" Face recognition and ViT emotion model loaded.")

# === SECTION 4: Match Video Analysis ===
st.subheader("🎥 Match Video Analysis")
video_files = {
    #"🇦🇷 Argentina vs France": "highlights/Argentina_France_Final_2022.mp4",
    "📼 Clip 1": "highlights/clip1.mp4",
    "📼 Clip 2": "highlights/clip2.mp4",
    "📼 Clip 3": "highlights/clip3.mp4",
    "📼 Clip 4": "highlights/clip4.mp4",
    "🧪 test": "highlights/testt.mp4"
}
selected_video = st.selectbox("📁 Choose Video", list(video_files.keys()))
video_path = video_files[selected_video]
fast_mode = st.checkbox("⚡ Fast Mode (process fewer frames)", value=True)
show_preview = st.checkbox("🎬 Show Video Preview", value=False)

def is_duplicate_ssim(frame1, frame2, threshold=0.99):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

if st.button("▶️ Run YOLO + MTCNN + Emotion Analysis"):
    if not os.path.exists(video_path):
        st.error("❌ Video file not found.")
        st.stop()

    st.session_state["emotion_analysis_done"] = False
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"🎞️ Total frames detected: {total_frames}")
    model = YOLO("yolov8n.pt")
    results = []
    seen_emotions = set()
    progress = st.progress(0)
    status = st.empty()
    frame_interval = 10 if fast_mode else 1
    prev_frame = None
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if i % frame_interval != 0:
            i += 1
            continue
        if prev_frame is not None and is_duplicate_ssim(prev_frame, frame):
            i += 1
            continue

        #status.text(f"⚪ Frame {i}/{total_frames}")
        progress.progress(i / total_frames)

        if show_preview and i % 30 == 0:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i}", use_container_width=True)

        detections = model(frame, verbose=False)
        for box in detections[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            person = frame[y1:y2, x1:x2]

            if person.size == 0:
                continue
            try:
                boxes, _ = mtcnn.detect(person)
                if boxes is None or len(boxes) == 0:
                    continue
            except:
                continue

            for fx1, fy1, fx2, fy2 in boxes:
                fx1, fy1 = max(0, int(fx1)), max(0, int(fy1))
                fx2, fy2 = min(person.shape[1], int(fx2)), min(person.shape[0], int(fy2))
                face = person[fy1:fy2, fx1:fx2]
                if face.size == 0:
                    continue

                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    enc = face_recognition.face_encodings(face_rgb)
                    if not enc:
                        continue
                    encoding = enc[0]
                    name = "Unknown"
                    min_dist = 0.6
                    for known_name, known_enc in all_faces.items():
                        dist = face_recognition.face_distance([known_enc], encoding)[0]
                        if dist < min_dist:
                            name = known_name
                            min_dist = dist
                except:
                    continue

                if name == "Unknown":
                    continue

                try:
                    face_pil = Image.fromarray(face_rgb)
                    inputs = feature_extractor(images=face_pil, return_tensors="pt")
                    with torch.no_grad():
                        outputs = vit_model(**inputs)
                    predicted_class_idx = outputs.logits.argmax(-1).item()
                    emotion = vit_model.config.id2label[predicted_class_idx]
                except:
                    emotion = "Unknown"

                key = (name, emotion)
                if key not in seen_emotions:
                    seen_emotions.add(key)
                    results.append({"player": name, "emotion": emotion})
                    cv2.imwrite(f"outputs/faces/frame_{i}_{name}.jpg", face)

        prev_frame = frame
        i += 1

    cap.release()
    df = pd.DataFrame(results)
    df.to_csv("outputs/emotion_results.csv", index=False)
    st.session_state["emotion_analysis_done"] = True
    st.success(f"✅ Done! {len(results)} unique emotions found.")
    st.dataframe(df)
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="emotion_results.csv")

# === SECTION 5: Player Stats & Injury Risk ===
st.subheader("📊 Player Stats & Injury Risk")

def clean_age(val):
    match = re.search(r"\d{2}", str(val))
    return int(match.group()) if match else 0

played = player_df[player_df["minutes_90s"] > 0].copy()
played["age"] = played["age"].apply(clean_age)
played["injury_risk_label"] = played["injury_risk"].map({1: "High", 0.5: "Medium", 0: "Low"})

player_names = sorted(played["player"].unique())
selected = st.selectbox("Select Player", player_names)

pdata = played[played["player"] == selected]

if not pdata.empty:
    st.metric("Fatigue Score", f"{pdata['fatigue_score'].values[0]:.2f}")
    st.warning(f"Risk Level: {pdata['injury_risk_label'].values[0]}")
    display_cols = ["player", "team", "club", "position", "age", "fatigue_score", "injury_risk_label"]
    st.dataframe(pdata[display_cols].reset_index(drop=True).T)

# === SECTION 6: Smart Personalized Training Plan ===
st.subheader("📋 Personalized Training Plan")

if st.session_state["emotion_analysis_done"]:
    emotion_path = "outputs/emotion_results.csv"
    if os.path.exists(emotion_path):
        emotion_df = pd.read_csv(emotion_path)

        # Deduplicate by player + emotion
        emotion_df = emotion_df.drop_duplicates(subset=["player", "emotion"])

        # Merge with main player stats
        player_info = player_df[[
            "player", "team", "club", "age", "position", "fatigue_score", "injury_risk"
        ]].copy()
        player_info["age"] = player_info["age"].apply(clean_age)
        player_info["injury_risk_label"] = player_info["injury_risk"].map({1: "High", 0.5: "Medium", 0: "Low"})

        def match_full_name(short_name, candidates):
            for full_name in candidates:
                if short_name.lower() in full_name.lower():
                    return full_name
            return short_name

        full_names = player_info["player"].tolist()
        emotion_df["matched_name"] = emotion_df["player"].apply(lambda x: match_full_name(x, full_names))
        merged_df = pd.merge(emotion_df, player_info, left_on="matched_name", right_on="player", how="left")

        def generate_plan(row):
            # Physical plan
            if row['injury_risk_label'] == "High":
                physical = "🧘 Recovery Day + Ice Baths"
            elif row['injury_risk_label'] == "Medium":
                physical = "🚶 Light Tactical Drills"
            else:
                physical = "🏃 High-Intensity Endurance Training"

            # Mental plan
            emo = str(row['emotion']).lower()
            if emo in ["angry", "sad", "fear"]:
                mental = "🧠 Mental Coaching & Motivation"
            elif emo == "happy":
                mental = " Match Readiness Confirmed"
            elif emo == "surprise":
                mental = "🔍 Monitor Reactions Closely"
            elif emo == "neutral":
                mental = " Calm – No Mental Action Needed"
            else:
                mental = "⚠️ Emotion Not Detected"

            return physical, mental

        merged_df[["physical_plan", "mental_plan"]] = merged_df.apply(generate_plan, axis=1, result_type="expand")

        final_plan = merged_df[[
            "matched_name", "fatigue_score", "injury_risk_label", "emotion", "physical_plan", "mental_plan"
        ]].rename(columns={"matched_name": "player"}).drop_duplicates()

        st.dataframe(final_plan)
        st.download_button("📥 Download Training Plan CSV", data=final_plan.to_csv(index=False),
                           file_name="smart_training_plan.csv", mime="text/csv")
    else:
        st.warning("Emotion CSV not found.")
else:
    st.info("Run the emotion analysis first to generate a personalized training plan.")

st.caption("Developed by Team AiWeb – Elaf & Fahad")