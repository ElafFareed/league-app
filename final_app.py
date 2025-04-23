# === IMPORTS ===
import streamlit as st
from pathlib import Path
import base64
import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import face_recognition
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from skimage.metrics import structural_similarity as ssim

def preprocess_player_df(df):
    if "fatigue_score" not in df.columns or "injury_risk" not in df.columns:
        features = ["minutes_90s", "tackles", "press", "blocks", "passes_completed", "miscontrols", "dispossessed"]
        features = [f for f in features if f in df.columns]
        if features:
            scaler = MinMaxScaler()
            df["fatigue_score"] = scaler.fit_transform(df[features]).mean(axis=1)
            df["injury_risk"] = df["fatigue_score"].apply(lambda x: 1 if x > 0.7 else (0.5 if x > 0.5 else 0))
        else:
            df["fatigue_score"] = 0
            df["injury_risk"] = 0

    if "age" in df.columns:
        df["age"] = df["age"].apply(lambda val: int(re.search(r"\d{2}", str(val)).group()
                                    if pd.notna(val) and re.search(r"\d{2}", str(val)) else 0))
    return df

# === PAGE SETUP ===
st.set_page_config(
    page_title="MatchMind AI Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM STYLING ===
st.markdown("""
    <style>
    .stButton > button {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# === CUSTOM BUTTON STYLING ===
st.markdown("""
    <style>
    .stButton > button {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df_stats = pd.read_csv("data/player_stats_cleaned.csv")
        df_misc = pd.read_csv("data/player_misc_cleaned.csv")
        df_playingtime = pd.read_csv("data/player_playingtime_cleaned.csv")
        df = df_stats.merge(df_misc, on=["player", "team"], how="outer")
        df = df.merge(df_playingtime, on=["player", "team"], how="outer")
        df.fillna(0, inplace=True)
        return df
    except:
        return pd.DataFrame()

player_df = preprocess_player_df(load_data())

if "fatigue_score" not in player_df.columns or "injury_risk" not in player_df.columns:
    features = ["minutes_90s", "tackles", "press", "blocks", "passes_completed", "miscontrols", "dispossessed"]
    features = [f for f in features if f in player_df.columns]

    if features:
        scaler = MinMaxScaler()
        player_df["fatigue_score"] = scaler.fit_transform(player_df[features]).mean(axis=1)
        player_df["injury_risk"] = player_df["fatigue_score"].apply(
            lambda x: 1 if x > 0.7 else (0.5 if x > 0.5 else 0))
    else:
        player_df["fatigue_score"] = 0
        player_df["injury_risk"] = 0

# === ENCODE BACKGROUND IMAGE ===
def img_to_base64(image_path):
    with open(image_path, "rb") as f:
        bg_data = f.read()
    return base64.b64encode(bg_data).decode()

try:
    bg_base64 = img_to_base64("ui/background.jpg")
except:
    bg_base64 = ""

# === STYLING ===
st.markdown(f"""
    <style>
    html, body {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        opacity: 0.98;
    }}

    /* === SIDEBAR STYLING === */
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-top-right-radius: 20px;
        border-bottom-right-radius: 20px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        font-family: "Times New Roman", serif;
        color: black;
        padding-left: 20px;
    }}

    [data-testid="stSidebar"] h3 {{
        color: black;
        margin-top: 10px;
        text-align: left;
    }}

    /* === BUTTONS === */
    button[kind="secondary"] {{
        background-color: rgba(255, 255, 255, 0.3);
        color: #222;
        font-family: "Times New Roman", serif;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 10px;
        padding: 10px 16px;
        border: none;
        box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.08);
        text-align: left;
    }}

    button[kind="secondary"]:hover {{
        background-color: rgba(255, 255, 255, 0.5);
    }}

    /* === MAIN CONTENT BOX === */
    .main-content {{
        background-color: transparent;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        color: white;
        font-family: "Times New Roman", serif;
    }}

    .metric-box {{
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }}

    .dataframe {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }}

    .stSelectbox, .stCheckbox, .stButton {{
        color: white !important;
    }}

    .stSelectbox div {{
        color: black !important;
    }}

    .stProgress > div > div > div {{
        background-color: #4CAF50 !important;
    }}

    /* === FIX FOR BLACK HEADER BAR === */
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0) !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }}

    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    /* === REMOVE TOP CONTAINER BACKGROUND === */
    [data-testid="stAppViewContainer"] > .main {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    </style>
""", unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
    <div style="
        position: fixed;
        bottom: 10px;
        left: 0;
        right: 0;
        text-align: center;
        color: white;
        font-size: 13px;
        font-family: 'Times New Roman', serif;
        opacity: 0.85;">
        AiWeb – Elaf & Fahad • MatchMind 2025
    </div>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if "section" not in st.session_state:
    st.session_state.section = "Video Analysis"
if "emotion_analysis_done" not in st.session_state:
    st.session_state["emotion_analysis_done"] = False
if "emotion_results" not in st.session_state:
    st.session_state["emotion_results"] = None

# === HELPER FUNCTIONS ===
def clean_age(val):
    match = re.search(r"\d{2}", str(val))
    return int(match.group()) if match else 0

def is_duplicate_ssim(frame1, frame2, threshold=0.99):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

def match_full_name(short_name, candidates):
    for full_name in candidates:
        if short_name.lower() in full_name.lower():
            return full_name
    return short_name

@st.cache_data
def load_wearable():
    try:
        return pd.read_csv("data/Simulated_Wearable_Argentina_France.csv")
    except Exception as e:
        st.error(f"Error loading wearable data: {e}")
        return pd.DataFrame()

def load_known_faces(folder):
    known = {}
    try:
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".png")):
                path = os.path.join(folder, filename)
                name = os.path.splitext(filename)[0]
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known[name] = encs[0]
    except Exception as e:
        st.error(f"Error loading faces from {folder}: {e}")
    return known

# === SIDEBAR ===
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>MatchMind AI</h3>", unsafe_allow_html=True)

    menu = {
        "Video Analysis": "🎥",
        "Injury Risk": "📊",
        "Personalized Training Plan": "📋",
        "Real-Time Wearable Risk Alerts": "🛡️"
    }

    for name, emoji in menu.items():
        if st.button(f"{emoji} {name}", key=f"menu_{name}"):
            st.session_state.section = name

# === MAIN CONTENT ===
with st.container():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)

    # === VIDEO ANALYSIS SECTION ===
    if st.session_state.section == "Video Analysis":
        st.header("🎥 Match Video Analysis")
       
        # Load face recognition models
        with st.spinner("Loading face recognition models..."):
            argentina_faces = load_known_faces("argentina")
            france_faces = load_known_faces("france")
            all_faces = {**argentina_faces, **france_faces}
            mtcnn = MTCNN(keep_all=True, device='cpu')
            model_name = "trpakov/vit-face-expression"
            vit_model = ViTForImageClassification.from_pretrained(model_name)
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
       
        video_files = {
            "📼 Argentina vs France 2022 World cup": "highlights/Argentina_France_Final_2022.mp4",
            "📼 Highlights": "highlights/argentina_france_highlight.mp4",
            "📼 Clip 1": "highlights/clip1.mp4",
            "📼 Clip 2": "highlights/clip2.mp4",
            "📼 Clip 3": "highlights/clip3.mp4",
            "🧪 Test Video": "highlights/testt.mp4"
        }
       
        col1, col2 = st.columns(2)
        with col1:
            selected_video = st.selectbox("Select Video", list(video_files.keys()))
        with col2:
            col_a, col_b = st.columns(2)
            with col_a:
                fast_mode = st.checkbox("Fast Mode", value=True, help="Process fewer frames for quicker analysis")
            with col_b:
                show_preview = st.checkbox("Show Preview", value=False, help="Display video frames during processing")
       
        video_path = video_files[selected_video]
       
        if st.button("▶️ Run Emotion Analysis", type="primary"):
            if not os.path.exists(video_path):
                st.error("❌ Video file not found.")
            else:
                st.session_state["emotion_analysis_done"] = False
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
               
                st.write(f"🎞️ Total frames: {total_frames}")
                model = YOLO("yolov8n.pt")
                results = []
                seen_emotions = set()
                progress_bar = st.progress(0)
                status_text = st.empty()
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

                    status_text.text(f"Processing frame {i}/{total_frames}")
                    progress_bar.progress(i / total_frames)

                    if show_preview and i % 30 == 0:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i}")

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
                                os.makedirs("outputs/faces", exist_ok=True)
                                cv2.imwrite(f"outputs/faces/frame_{i}_{name}.jpg", face)

                    prev_frame = frame
                    i += 1

                cap.release()
                df = pd.DataFrame(results)
                st.session_state["emotion_results"] = df
                st.session_state["emotion_analysis_done"] = True
                st.success(f"✅ Analysis complete! {len(results)} unique emotions detected.")
               
                if not df.empty:
                    st.dataframe(df)
                    st.download_button(
                        "📥 Download Results",
                        data=df.to_csv(index=False),
                        file_name="emotion_results.csv",
                        mime="text/csv"
                    )

    # === INJURY RISK SECTION ===
    elif st.session_state.section == "Injury Risk":
        st.header("📊 Player Stats & Injury Risk")
       
        player_df = preprocess_player_df(load_data())
        if player_df.empty:
            st.warning("No player data available. Please check your data files.")
        else:
            features = ["minutes_90s", "tackles", "press", "blocks", "passes_completed", "miscontrols", "dispossessed"]
            features = [f for f in features if f in player_df.columns]
            scaler = MinMaxScaler()
            player_df["fatigue_score"] = scaler.fit_transform(player_df[features]).mean(axis=1)
            player_df["injury_risk"] = player_df["fatigue_score"].apply(
                lambda x: 1 if x > 0.7 else (0.5 if x > 0.5 else 0))
           
            played = player_df[player_df["minutes_90s"] > 0].copy()
            played["age"] = played["age"].apply(clean_age)
            played["injury_risk_label"] = played["injury_risk"].map(
                {1: "High", 0.5: "Medium", 0: "Low"})
           
            selected = st.selectbox("Select Player", sorted(played["player"].unique()))
            pdata = played[played["player"] == selected]
           
            if not pdata.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class="metric-box">
                            <h3>Fatigue Score</h3>
                            <h2>{pdata['fatigue_score'].values[0]:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    risk = pdata['injury_risk_label'].values[0]
                    risk_color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
                    st.markdown(f"""
                        <div class="metric-box">
                            <h3>Risk Level</h3>
                            <h2 style="color: {risk_color}">{risk}</h2>
                        </div>
                    """, unsafe_allow_html=True)
               
                display_cols = ["player", "team", "club", "position", "age", "fatigue_score", "injury_risk_label"]
                st.dataframe(pdata[display_cols].reset_index(drop=True).T)

    # === TRAINING PLAN SECTION ===
    elif st.session_state.section == "Personalized Training Plan":
        st.header("📋 Personalized Training Plan")
       
        if st.session_state["emotion_analysis_done"] and st.session_state["emotion_results"] is not None:
            emotion_df = st.session_state["emotion_results"]
            player_df = preprocess_player_df(load_data())
           
            if not player_df.empty and not emotion_df.empty:
                # Deduplicate by player + emotion
                emotion_df = emotion_df.drop_duplicates(subset=["player", "emotion"])

                # Merge with main player stats
                player_info = player_df[[
                    "player", "team", "club", "age", "position", "fatigue_score", "injury_risk"
                ]].copy()
                player_info["age"] = player_info["age"].apply(clean_age)
                player_info["injury_risk_label"] = player_info["injury_risk"].map(
                    {1: "High", 0.5: "Medium", 0: "Low"})

                full_names = player_info["player"].tolist()
                emotion_df["matched_name"] = emotion_df["player"].apply(
                    lambda x: match_full_name(x, full_names))
                merged_df = pd.merge(
                    emotion_df,
                    player_info,
                    left_on="matched_name",
                    right_on="player",
                    how="left"
                )

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
                        mental = "✅ Match Readiness Confirmed"
                    elif emo == "surprise":
                        mental = "🔍 Monitor Reactions Closely"
                    elif emo == "neutral":
                        mental = "😐 Calm – No Action Needed"
                    else:
                        mental = "⚠️ Emotion Not Detected"

                    return physical, mental

                merged_df[["physical_plan", "mental_plan"]] = merged_df.apply(
                    generate_plan, axis=1, result_type="expand")

                final_plan = merged_df[[
                    "matched_name",
                    "emotion", "physical_plan", "mental_plan"
                ]].rename(columns={"matched_name": "player"}).drop_duplicates()

                st.dataframe(final_plan)
                st.download_button(
                    "📥 Download Training Plan",
                    data=final_plan.to_csv(index=False),
                    file_name="smart_training_plan.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Player data or emotion results not available.")
        else:
            st.info("ℹ️ Please run the emotion analysis first to generate training plans.")

    # === WEARABLE ALERTS SECTION ===
    elif st.session_state.section == "Real-Time Wearable Risk Alerts":
        st.header("🛡️ Real-Time Wearable Risk Alerts")
       
        df_wearable = load_wearable()
       
        if st.session_state["emotion_analysis_done"] and st.session_state["emotion_results"] is not None:
            df_emotion = st.session_state["emotion_results"]
            df_emotion = df_emotion.drop_duplicates(subset=["player", "emotion"])
            alerts = []

            def check_wearable_risk(player, emotion):
                row = df_wearable[df_wearable["player_name"] == player].sort_values("timestamp").tail(1)
                if row.empty:
                    return None

                hr = row["heart_rate"].values[0]
                motion = row["motion_level"].values[0]
                emg = row["emg_signal"].values[0]
                fatigue = row["fatigue_score"].values[0]

                risky_emotion = str(emotion).lower() in ["tired", "sad", "angry"]
                if risky_emotion and hr > 170 and motion < 10 and emg < 0.4 and fatigue > 0.7:
                    return f"⚠️ High Injury Risk – Recommend substitution for {player}"

                return None

            for _, row in df_emotion.iterrows():
                alert = check_wearable_risk(row["player"], row["emotion"])
                if alert:
                    alerts.append(alert)
                    st.error(alert)

            if not alerts:
                st.success("✅ No high-risk players detected based on wearable data.")
        else:
            st.info("ℹ️ Please run the emotion analysis first to check for risk alerts.")

    st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
    <div style="
        position: fixed;
        bottom: 10px;
        left: 0;
        right: 0;
        text-align: center;
        color: white;
        font-size: 13px;
        font-family: 'Times New Roman', serif;
        opacity: 0.85;">
        AiWeb – Elaf & Fahad • MatchMind 2025
    </div>
""", unsafe_allow_html=True)