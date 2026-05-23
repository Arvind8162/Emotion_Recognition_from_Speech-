# Bootstrap dependencies that have incompatible PyPI metadata (dienen/paips require tensorflow==2.4.1)
import subprocess
import sys
import os

try:
    import dienen
    import paips
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dienen==0.0.3", "paips==0.0.5", "--no-deps"])

import streamlit as st
import joblib
import numpy as np
import librosa
import torch
import fairseq
import time
import io
import tempfile
import pandas as pd
import soundfile as sf

st.set_page_config(page_title="Speech Emotion AI", layout="wide", initial_sidebar_state="collapsed")

# ─── Premium UI/UX Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Premium Dark Background with Glowing Orbs */
    .stApp {
        background-color: #0B0E14 !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(255, 107, 107, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(78, 205, 196, 0.08) 0%, transparent 50%);
        background-attachment: fixed;
        color: #E2E8F0;
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #FFF 0%, #A0ABC0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fade-in-up 0.8s ease-out;
    }
    
    .hero-title .highlight {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse-glow 3s infinite alternate;
    }
    
    .hero-sub {
        text-align: center;
        font-size: 1.2rem;
        color: #8B9BB4;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fade-in-up 1s ease-out;
    }
    
    /* Result Card (Glassmorphism) */
    .result-card {
        background: rgba(20, 25, 35, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
        animation: slide-up 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2);
    }
    
    .emotion-emoji {
        font-size: 5.5rem;
        display: inline-block;
        margin-bottom: 15px;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.3));
        animation: bounce 2s infinite ease-in-out;
    }
    
    .emotion-name {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .confidence-text {
        color: #A0ABC0;
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Info/Alert Boxes */
    .info-box {
        background: rgba(78, 205, 196, 0.05);
        border-left: 4px solid #4ECDC4;
        border-radius: 8px;
        padding: 16px 20px;
        color: #A0ABC0;
        font-size: 0.95rem;
        margin: 15px 0 25px 0;
        display: flex;
        align-items: center;
        animation: fade-in 1s ease-out;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(20, 25, 35, 0.5);
        border-radius: 16px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 24px;
        color: #8B9BB4;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFF;
        background: rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1)) !important;
        color: #FFF !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #ee5a24 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 30px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 15px 25px rgba(255, 107, 107, 0.4) !important;
        background: linear-gradient(135deg, #ff7e7e 0%, #f36c39 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 5px 10px rgba(255, 107, 107, 0.3) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: rgba(20, 25, 35, 0.5) !important;
        border: 2px dashed rgba(78, 205, 196, 0.4) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader > div > div:hover {
        background: rgba(78, 205, 196, 0.05) !important;
        border-color: #4ECDC4 !important;
    }
    
    /* Audio Player */
    stAudio {
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: rgba(20, 25, 35, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(78, 205, 196, 0.3);
    }
    
    /* Animations */
    @keyframes fade-in-up {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes slide-up {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes pulse-glow {
        0% { filter: drop-shadow(0 0 15px rgba(255, 107, 107, 0.4)); }
        100% { filter: drop-shadow(0 0 25px rgba(78, 205, 196, 0.6)); }
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 4rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎙️ Speech Emotion <span class="highlight">AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload audio or speak live to detect 7 distinct emotions using deep learning.</div>', unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Check for local path first to save download time/bandwidth
    local_dev_path = r'C:\Users\arvin\Models\wav2vec2\wav2vec_small.pt'
    if os.path.exists(local_dev_path):
        model_path = local_dev_path
    else:
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'wav2vec_small.pt')
        
        # Create directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Download model if missing (required for hosting since the file is >900MB and not in Git)
        if not os.path.exists(model_path):
            import requests
            with st.status("📥 Downloading Wav2Vec2 Base Model (900MB)... This may take a few minutes.", expanded=True) as status:
                url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
                response = requests.get(url, stream=True)
                with open(model_path, 'wb') as f:
                    for data in response.iter_content(chunk_size=1024*1024):
                        f.write(data)
                status.update(label="✅ Download Complete!", state="complete")

    w2v_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    w2v_model = w2v_model[0]
    w2v_model.eval()
    
    clf_path = os.path.join('experiments', 'run', 'MainTask', 'DownstreamRavdess', 'RavdessModel', 'out')
    stats_path = os.path.join('experiments', 'run', 'MainTask', 'DownstreamRavdess', 'RavdessStatisticsTrain', 'out')
    
    clf_model = joblib.load(clf_path)
    stats = joblib.load(stats_path)
    
    return w2v_model, clf_model, stats

with st.spinner("⏳ Loading AI Models..."):
    try:
        w2v_model, clf_model, stats = load_models()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# ─── Config ──────────────────────────────────────────────────────────────────
emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
emotion_emojis = {'angry':'😡','disgust':'🤢','fearful':'😨','happy':'😊','neutral':'😐','sad':'😢','surprised':'😲'}

# ─── Core Prediction Function ───────────────────────────────────────────────
def predict_emotion_from_numpy(audio_np, sr):
    """Takes a numpy audio array and sample rate, returns (probabilities, emotion_name)."""
    
    if len(audio_np) == 0:
        return None, None
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    
    # Truncate to 4 seconds max
    max_samples = 64000
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]
    
    x_tensor = torch.tensor(np.expand_dims(audio_np, axis=0).astype(np.float32))
    
    # Extract Wav2Vec2 features
    with torch.no_grad():
        activations = w2v_model.extract_features(x_tensor, None)
        features = activations[0].cpu().numpy()       # (Time, Batch, 512)
        features = np.transpose(features, (1, 0, 2))  # (Batch, Time, 512)
    
    # Normalize
    actor_stats = stats['wav2vec2']['actor']
    all_means = [actor_stats[a]['mean'] for a in actor_stats]
    all_stds = [actor_stats[a]['std'] for a in actor_stats]
    global_mean = np.mean(all_means, axis=0)
    global_std = np.mean(all_stds, axis=0)
    features = (features - global_mean) / (global_std + 1e-8)
    
    # Pad or truncate to 250 frames
    MAX_FRAMES = 250
    num_frames = features.shape[1]
    if num_frames > MAX_FRAMES:
        features = features[:, :MAX_FRAMES, :]
        mask = np.ones((1, MAX_FRAMES), dtype=np.float32)
    else:
        pad_len = MAX_FRAMES - num_frames
        features = np.pad(features, ((0,0),(0,pad_len),(0,0)), mode='constant')
        mask = np.zeros((1, MAX_FRAMES), dtype=np.float32)
        mask[:, :num_frames] = 1.0
    
    pred_probs = clf_model.predict({'input_signal': features, 'signal_mask': mask})[0]
    return pred_probs, emotion_labels[np.argmax(pred_probs)]


def load_audio_from_bytes(audio_bytes):
    """Load audio bytes into numpy array using soundfile, fallback to librosa."""
    try:
        # Try soundfile first (handles WAV well)
        buf = io.BytesIO(audio_bytes)
        audio_np, sr = sf.read(buf, dtype='float32')
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)  # Convert stereo to mono
        return audio_np, sr
    except Exception:
        pass
    
    # Fallback: save to temp file and use librosa
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio_np, sr = librosa.load(tmp_path, sr=None)
        return audio_np, sr
    finally:
        os.unlink(tmp_path)


def display_results(pred_probs, predicted_emotion):
    """Show beautiful prediction results."""
    emoji = emotion_emojis[predicted_emotion]
    confidence = pred_probs[np.argmax(pred_probs)] * 100
    
    st.markdown(f"""
    <div class="result-card">
        <span class="emotion-emoji">{emoji}</span>
        <p class="emotion-name">{predicted_emotion.capitalize()}</p>
        <p class="confidence-text">Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📊 Emotion Probabilities")
    chart_data = pd.DataFrame(
        pred_probs * 100,
        index=[f"{emotion_emojis[e]} {e.capitalize()}" for e in emotion_labels],
        columns=['Confidence (%)']
    )
    st.bar_chart(chart_data)
    
    cols = st.columns(len(emotion_labels))
    for i, (col, label) in enumerate(zip(cols, emotion_labels)):
        with col:
            st.metric(label=emotion_emojis[label], value=f"{pred_probs[i]*100:.1f}%", delta=label.capitalize())


def run_prediction(audio_bytes):
    """Full pipeline: bytes -> load -> predict -> display."""
    audio_np, sr = load_audio_from_bytes(audio_bytes)
    
    if audio_np is None or len(audio_np) == 0:
        st.warning("⚠️ Audio is empty. Please try again.")
        return
    
    # Re-encode to proper WAV for playback
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_np, sr, format='WAV', subtype='PCM_16')
    wav_buf.seek(0)
    st.audio(wav_buf, format='audio/wav')
    
    pred_probs, predicted_emotion = predict_emotion_from_numpy(audio_np, sr)
    
    if pred_probs is not None:
        display_results(pred_probs, predicted_emotion)
    else:
        st.warning("⚠️ Could not analyze audio.")


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record Live Voice"])

# ═══ Tab 1: Upload ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="info-box">💡 Supports: WAV, MP3, M4A, OGG — Upload any speech recording to detect emotion</div>', unsafe_allow_html=True)
    
    _, col_center, _ = st.columns([1, 3, 1])
    with col_center:
        uploaded_file = st.file_uploader("Drag & drop your audio file", type=['wav','mp3','m4a','ogg'], key="upload")
    
    if uploaded_file is not None:
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("🔮 Analyze Emotion", use_container_width=True, key="btn_upload"):
                with st.spinner("🧠 Analyzing..."):
                    try:
                        run_prediction(uploaded_file.read())
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ═══ Tab 2: Record ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="info-box">🎙️ Click the mic icon to record. Speak for 2-4 seconds. Click again to stop. Then click "Analyze" to predict emotion!</div>', unsafe_allow_html=True)
    
    from streamlit_mic_recorder import mic_recorder
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        audio = mic_recorder(
            start_prompt="🔴 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key='mic_recorder'
        )
    
    if audio is not None:
        # Re-encode the audio for proper playback
        try:
            raw_bytes = audio['bytes']
            audio_np, sr = load_audio_from_bytes(raw_bytes)
            
            if audio_np is not None and len(audio_np) > 0:
                # Create proper WAV for playback
                playback_buf = io.BytesIO()
                sf.write(playback_buf, audio_np, sr, format='WAV', subtype='PCM_16')
                playback_buf.seek(0)
                
                st.markdown("**🔊 Your Recording:**")
                st.audio(playback_buf, format='audio/wav')
                
                _, col_btn, _ = st.columns([1, 2, 1])
                with col_btn:
                    if st.button("🔮 Analyze My Voice", use_container_width=True, key="btn_record"):
                        with st.spinner("🧠 Analyzing your voice..."):
                            try:
                                pred_probs, predicted_emotion = predict_emotion_from_numpy(audio_np, sr)
                                if pred_probs is not None:
                                    display_results(pred_probs, predicted_emotion)
                                else:
                                    st.warning("⚠️ Could not detect speech.")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Recording was empty. Please try again.")
        except Exception as e:
            st.error(f"❌ Could not process recording: {str(e)}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-text">Built with ❤️ using Wav2Vec2 (Facebook AI) + Dienen + Streamlit | Trained on RAVDESS Dataset</div>', unsafe_allow_html=True)
