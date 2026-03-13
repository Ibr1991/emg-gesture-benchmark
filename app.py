import streamlit as st
import time
import random
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64

# =========================================
# 1. PAGE CONFIG & ADVANCED CSS (updated)
# =========================================
st.set_page_config(page_title="EMG Gesture Benchmark", layout="wide")

st.markdown("""
<style>
body { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.main { background-color: rgba(255,255,255,0.97); padding: 2rem; border-radius: 12px; }

/* Grid Card Styling */
[data-testid="column"] { display: flex; flex-direction: column; align-items: center; justify-content: center; }

/* -------- gesture column alignment helpers -------- */
[data-testid="column"] {
    /* keep current centering but enforce consistent column height for gesture grid */
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start; /* stack top->bottom so we can push button to bottom */
    min-height: 170px;            /* same height for all gesture columns */
}

/* make the image container a fixed area so images don't force different heights */
.stImage {
    height: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* push button to the bottom of the column so all buttons align horizontally */
div.stButton {
    margin-top: auto; /* this pushes the button down inside the column */
    width: 100%;
    display: flex;
    justify-content: center;
}
/* Image and Button Width Consistency
   Make buttons inline so columns keep them horizontally aligned */
.stImage > img { 
    width: 100% !important; 
    max-width: 100px !important; 
    height: 100px !important; 
    object-fit: contain; 
    border-radius: 8px; 
    background: #f0f2f6; 
    padding: 5px; 
}

/* Allow horizontal buttons in columns (no forced full width) */
div.stButton > button { 
    width: auto !important; 
    min-width: 90px !important;
    height: 40px !important; 
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem !important; 
    background-color: #1f77b4 !important; 
    color: white !important;
    padding: 4px 8px !important;
    margin-top: 6px;
}

/* Sidebar Button Fix */
section[data-testid="stSidebar"] div.stButton > button {
    width: auto !important;
    height: auto !important;
    padding: 10px 20px !important;
    font-size: 0.9rem !important;
}

/* Logic Blocks (UNIFIED default blue) */
.logic-block { 
    width: 100%; height: 72px; display: flex; align-items: center; justify-content: center; 
    text-align: center; border-radius: 8px; font-weight: bold; font-size: 0.95rem; 
    color: white; border: 2px solid transparent; padding: 5px;
    background-color: #2980b9; border-color: #3498db;
}

/* Individual block type classes kept for semantic clarity (but same base blue) */
.block-raw { background-color: #2980b9; border-color: #3498db; }
.block-filter { background-color: #2980b9; border-color: #3498db; }
.block-window { background-color: #2980b9; border-color: #3498db; }
.block-feature { background-color: #2980b9; border-color: #3498db; }
.block-ai { background-color: #2980b9; border-color: #3498db; }

/* Done state (green) */
.block-done { background-color: #2ecc71 !important; border-color: #27ae60 !important; color: #062b16 !important; }

/* Active state */
.active-block { transform: scale(1.03); box-shadow: 0 0 12px rgba(0,0,0,0.25); }

/* Animations */
.pumping-brain { animation: pump 0.6s infinite; }
@keyframes pump { 0% { transform: scale(1); } 50% { transform: scale(1.08); } 100% { transform: scale(1); } }
@keyframes pulse-red { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.2); opacity: 0.8; } 100% { transform: scale(1); opacity: 1; } }
.pulsing { animation: pulse-red 0.8s infinite; }

/* Pipeline Aesthetics */
.arm-outer-wrapper { display: flex; justify-content: center; align-items: center; height: 300px; }
.arm-container { position: relative; height: 100%; width: auto; }
.arm-container img { height: 100%; width: auto; border-radius: 10px; }
.electrode-dot { position: absolute; width: 10px; height: 10px; background-color: #ff4b4b; border-radius: 50%; border: 1.5px solid white; box-shadow: 0 0 8px #ff4b4b; z-index: 99; }
.value-text { font-family: monospace; color: #2ecc71; font-size: 0.75rem; line-height: 1.1; white-space: pre-line; }
</style>
""", unsafe_allow_html=True)

# =========================================
# 2. UTILITY & SESSION STATE
# =========================================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

img_base64 = get_base64_of_bin_file("hand.png")
#----- Load .csv file
@st.cache_data
def load_benchmark_data():
    # Reads the CSV once and stores it in memory for fast lookup
    return pd.read_csv("benchmark_metrics.csv")

# Load the master database into a variable we can use later
master_db = load_benchmark_data()
# -----------------------------------

if "results_tables" not in st.session_state:
    st.session_state.results_tables = {
        "Dataset 1": pd.DataFrame(columns=["Model", "Input Gesture", "Latency (ms)", "RAM (KB)", "Energy (mJ)", "Predicted Gesture"]),
        "Dataset 2": pd.DataFrame(columns=["Model", "Input Gesture", "Latency (ms)", "RAM (KB)", "Energy (mJ)", "Predicted Gesture"])
    }
if "last_run_gesture" not in st.session_state:
    st.session_state.last_run_gesture = None
if "animation_running" not in st.session_state:
    st.session_state.animation_running = False
# =========================================
# 3. SIDEBAR & DATA
# =========================================
st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])

if "current_ds" not in st.session_state: st.session_state.current_ds = dataset_choice
if dataset_choice != st.session_state.current_ds:
    st.session_state.current_ds = dataset_choice
    st.session_state.last_run_gesture = None
    st.rerun()

model_choice = st.sidebar.selectbox("Choose Algorithm", ["Tsetlin Machine", "Random Forest", "1D CNN", "MLP", "kNN", "BNN"])

if dataset_choice == "Dataset 1":
    gestures = [("Up", "images/d1/1.PNG"), ("Down", "images/d1/2.PNG"), ("Left", "images/d1/3.PNG"), ("Right", "images/d1/4.PNG"), ("Index Point", "images/d1/5.PNG"), ("Two Finger Pinch", "images/d1/6.PNG"), ("Power Grasp", "images/d1/7.PNG"), ("Middle Finger Pinch", "images/d1/8.PNG"), ("Splay", "images/d1/9.PNG"), ("Index Finger Pinch", "images/d1/10.PNG")]
else:
    names = ["Thumb up","Two fingers","Three fingers","Four fingers","Five fingers","Fist","Pointing index","Adduction fingers","Wrist supination_1","Wrist pronation_1","Wrist supination_2","Wrist pronation_2","Wrist flexion","Wrist Extension","Wrist deviation_1","Wrist deviation_2","Wrist extension_2","Large grasp","Small grasp","Hook grasp","Index finger grasp","Medium wrap","Ring grasp","Four fingers grasp","Stick grasp","Writing grasp","Power sphere","Three fingers grasp","Precision grasp","Tripod grasp","Prismatic grasp","Tip grasp","Quadpod grasp","Lateral grasp","extension grasp","type grasp","disk grasp","Open grasp","Turn a screw","Cut something"]
    gestures = [(n, f"images/d2/{i}.PNG") for i, n in enumerate(names, 1)]

if st.sidebar.button(f"🗑️ Clear {dataset_choice} & Animation"):
    st.session_state.results_tables[dataset_choice] = pd.DataFrame(
    columns=["Model", "Input Gesture", "Latency (ms)", "RAM (KB)", "Energy (mJ)", "Predicted Gesture"]
    )
    st.session_state.last_run_gesture = None
    st.rerun()

# =========================================
# 4. ANIMATION ENGINE (EDITED)
# =========================================

# storage for frozen snapshots
if "stage_snapshots" not in st.session_state:
    st.session_state.stage_snapshots = {
        "raw": None,
        "filter": None,
        "window": None,
        "features": None,
        "gesture": None
    }

def display_electrode_arm(is_active=False):
    pulse_class = "pulsing" if is_active else ""
    coords = [(27, 30), (30, 45), (30, 65), (27, 80), (60, 45), (60, 60)]
    electrode_html = "".join(
        [f'<div class="electrode-dot {pulse_class}" style="top:{y}%; left:{x}%;"></div>' for y, x in coords]
    )

    st.markdown(
        f'<div class="arm-outer-wrapper"><div class="arm-container">'
        f'<img src="data:image/png;base64,{img_base64}">{electrode_html}</div></div>',
        unsafe_allow_html=True
    )


def run_inference_animation(gesture_name):

    st.session_state.last_run_gesture = gesture_name
    gesture_path = next((p for n, p in gestures if n == gesture_name), None)

    with animation_canvas.container(border=True):

        st.markdown(f"### 🛡️ Real-Time Inference: **{gesture_name}**")

        cols = st.columns([1.2, 1.4, 1.4, 1.4, 1.2, 1.0, 1.2])

        p_arm = cols[0].empty()

        raw_col = cols[1].container()
        filter_col = cols[2].container()
        window_col = cols[3].container()
        feature_col = cols[4].container()
        ai_col = cols[5].container()
        p_res = cols[6].empty()

        with raw_col:
            p_raw_block = st.empty()
            p_raw_plot = st.empty()

        with filter_col:
            p_filter_block = st.empty()
            p_filter_plot = st.empty()

        with window_col:
            p_window_block = st.empty()
            p_window_plot = st.empty()

        with feature_col:
            p_feature_block = st.empty()
            p_feature_vals = st.empty()

        with ai_col:
            p_ai_block = st.empty()

        stages = ["Raw EMG data", "Bandpass Filter", "Windowing", "Feature Extraction", "AI Model"]

        f_names = ["MAV","RMS","WL","SSC","ZC","VAR","WA","LOG","DAMV","MHW","IAV","MYOP","WAMP","SSI"]

        total_steps_per_stage = 7
        steps = len(stages) * total_steps_per_stage

        def _block_classes(stage_idx, current_active, base_class):

            classes = f"logic-block {base_class}"

            if current_active == stage_idx:
                classes += " active-block"

            if current_active > stage_idx:
                classes += " block-done"

            return classes

        raw_snapshot = None
        filter_snapshot = None
        window_snapshot = None
        feature_snapshot = None

        for step in range(steps):

            active_stage = step // total_steps_per_stage

            with p_arm:
                display_electrode_arm(is_active=True)

            # ==========================
            # RAW STAGE
            # ==========================
            raw_classes = _block_classes(0, active_stage, "block-raw")
            p_raw_block.markdown(f'<div class="{raw_classes}">RAW EMG DATA</div>', unsafe_allow_html=True)

            if active_stage == 0:

                raw = np.random.normal(0,0.9,(60,12))

                raw_snapshot = raw

                fig = go.Figure()

                for i in range(12):
                    fig.add_trace(go.Scatter(y=raw[:,i] + (i*1.5),mode="lines",line=dict(width=1)))

                fig.update_layout(height=170,margin=dict(l=0,r=0,t=0,b=0),
                showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

                p_raw_plot.plotly_chart(fig,use_container_width=True)

            else:
                p_raw_plot.empty()

            # ==========================
            # FILTER STAGE
            # ==========================
            filter_classes = _block_classes(1, active_stage, "block-filter")
            p_filter_block.markdown(f'<div class="{filter_classes}">BANDPASS FILTER</div>', unsafe_allow_html=True)

            if active_stage == 1:

                smooth = np.random.normal(0,0.25,(60,12))

                filter_snapshot = smooth

                fig = go.Figure()

                for i in range(12):
                    fig.add_trace(go.Scatter(y=smooth[:,i] + (i*1.5),mode="lines",line=dict(width=1)))

                fig.update_layout(height=170,margin=dict(l=0,r=0,t=0,b=0),
                showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

                p_filter_plot.plotly_chart(fig,use_container_width=True)

            else:
                p_filter_plot.empty()

            # ==========================
            # WINDOW STAGE
            # ==========================
            window_classes = _block_classes(2, active_stage, "block-window")
            p_window_block.markdown(f'<div class="{window_classes}">WINDOWING</div>', unsafe_allow_html=True)

            if active_stage == 2:

                window_data = np.random.normal(0,0.2,(60,12))

                window_snapshot = window_data

                fig = go.Figure()

                for i in range(12):
                    fig.add_trace(go.Scatter(y=window_data[:,i] + (i*1.5),mode="lines",line=dict(width=1)))

                for lp in [15,30,45]:
                    fig.add_vline(x=lp,line_width=2,line_color="black")

                fig.update_layout(height=170,margin=dict(l=0,r=0,t=0,b=0),
                showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

                p_window_plot.plotly_chart(fig,use_container_width=True)

            else:
                p_window_plot.empty()

            # ==========================
            # FEATURE STAGE
            # ==========================
            feature_classes = _block_classes(3, active_stage, "block-feature")
            p_feature_block.markdown(f'<div class="{feature_classes}">FEATURE EXTRACTION</div>', unsafe_allow_html=True)

            if active_stage == 3:

                feature_values = {f:random.random() for f in f_names}

                feature_snapshot = feature_values

                f_html = "".join([f"{k}: {v:.3f}<br>" for k,v in feature_values.items()])

                p_feature_vals.markdown(f'<div class="value-text">{f_html}</div>',unsafe_allow_html=True)

            else:
                p_feature_vals.empty()

            # ==========================
            # AI STAGE
            # ==========================
            ai_classes = _block_classes(4, active_stage, "block-ai")
            pumping = "pumping-brain" if active_stage == 4 else ""

            p_ai_block.markdown(
                f'<div class="{ai_classes} {pumping}">🧠 AI MODEL</div>',
                unsafe_allow_html=True
            )

            time.sleep(0.12)

        # store frozen snapshots
        st.session_state.stage_snapshots["raw"] = raw_snapshot
        st.session_state.stage_snapshots["filter"] = filter_snapshot
        st.session_state.stage_snapshots["window"] = window_snapshot
        st.session_state.stage_snapshots["features"] = feature_snapshot
        st.session_state.stage_snapshots["gesture"] = gesture_path

        if gesture_path and os.path.exists(gesture_path):
            p_res.image(gesture_path,width=120)
        else:
            p_res.image("https://via.placeholder.com/120",width=120)

        p_res.success(f"Classification Result:\n\n**{gesture_name}**")

        return time.perf_counter()
# =========================================
# 5. MAIN LAYOUT
# =========================================
st.title("EMG Gesture Benchmark Platform")

# GESTURE GRID (Images + horizontally aligned Buttons)
st.subheader(f"Select Gesture ({dataset_choice})")
cols_per_row = 10 
chunks = [gestures[i:i + cols_per_row] for i in range(0, len(gestures), cols_per_row)]

for chunk in chunks:
    cols = st.columns(cols_per_row)
    # Put image and button vertically stacked inside each column so buttons remain on a single horizontal row
    for i, (g_name, g_path) in enumerate(chunk):
        with cols[i]:
            if os.path.exists(g_path):
                st.image(g_path, use_container_width=False, width=100)
            else:
                st.image("https://via.placeholder.com/100", width=100)
            # button directly under image; thanks to CSS it's inline/auto width and will align horizontally across columns
            if st.button(g_name, key=f"btn_{dataset_choice}_{g_name}"):
                st.session_state.pending_run = g_name

animation_canvas = st.empty()
# PERSISTENT HUB (FINAL FIXED VERSION)

# PERSISTENT HUB (FINAL FIXED VERSION)

if st.session_state.last_run_gesture and not st.session_state.animation_running and "pending_run" not in st.session_state:

    with animation_canvas.container(border=True):
        
        # ---> THE FIX: We add an extra container to break Streamlit's component cache! <---
        with st.container():

            st.markdown(f"### 🛡️ Analysis Complete: **{st.session_state.last_run_gesture}**")

            compact_cols = st.columns([1.0,1.2,1.2,1.2,1.0,1.0,1.4])

            # ARM
            with compact_cols[0]:
                display_electrode_arm(is_active=False)

            # ======================
            # RAW
            # ======================
            with compact_cols[1]:

                st.markdown(
                    '<div class="logic-block block-raw block-done">RAW EMG DATA</div>',
                    unsafe_allow_html=True
                )

                raw = st.session_state.stage_snapshots["raw"]

                if raw is not None:

                    fig = go.Figure()

                    for i in range(12):
                        fig.add_trace(go.Scatter(
                            y=raw[:,i] + (i*1.5),
                            mode='lines',
                            line=dict(width=1)
                        ))

                    fig.update_layout(
                        height=170,
                        margin=dict(l=0,r=0,t=0,b=0),
                        showlegend=False,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )

                    st.plotly_chart(fig,use_container_width=True)


            # ======================
            # FILTER
            # ======================
            with compact_cols[2]:

                st.markdown(
                    '<div class="logic-block block-filter block-done">BANDPASS FILTER</div>',
                    unsafe_allow_html=True
                )

                filt = st.session_state.stage_snapshots["filter"]

                if filt is not None:

                    fig = go.Figure()

                    for i in range(12):
                        fig.add_trace(go.Scatter(
                            y=filt[:,i] + (i*1.5),
                            mode='lines',
                            line=dict(width=1)
                        ))

                    fig.update_layout(
                        height=170,
                        margin=dict(l=0,r=0,t=0,b=0),
                        showlegend=False,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )

                    st.plotly_chart(fig,use_container_width=True)


            # ======================
            # WINDOW
            # ======================
            with compact_cols[3]:

                st.markdown(
                    '<div class="logic-block block-window block-done">WINDOWING</div>',
                    unsafe_allow_html=True
                )

                win = st.session_state.stage_snapshots["window"]

                if win is not None:

                    fig = go.Figure()

                    for i in range(12):
                        fig.add_trace(go.Scatter(
                            y=win[:,i] + (i*1.5),
                            mode='lines',
                            line=dict(width=1)
                        ))

                    for lp in [15,30,45]:
                        fig.add_vline(x=lp,line_width=2,line_color="black")

                    fig.update_layout(
                        height=170,
                        margin=dict(l=0,r=0,t=0,b=0),
                        showlegend=False,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )

                    st.plotly_chart(fig,use_container_width=True)


            # ======================
            # FEATURES
            # ======================
            with compact_cols[4]:

                st.markdown(
                    '<div class="logic-block block-feature block-done">FEATURE EXTRACTION</div>',
                    unsafe_allow_html=True
                )

                feats = st.session_state.stage_snapshots["features"]

                if feats:

                    f_html = "".join([f"{k}: {v:.3f}<br>" for k,v in feats.items()])

                    st.markdown(
                        f'<div class="value-text">{f_html}</div>',
                        unsafe_allow_html=True
                    )


            # ======================
            # AI BLOCK
            # ======================
            with compact_cols[5]:

                st.markdown(
                    '<div class="logic-block block-ai block-done">🧠 AI MODEL</div>',
                    unsafe_allow_html=True
                )


            # ======================
            # RESULT
            # ======================
            with compact_cols[6]:

                gesture_path = st.session_state.stage_snapshots["gesture"]

                if gesture_path and os.path.exists(gesture_path):
                    st.image(gesture_path,width=110)
                else:
                    st.image("https://via.placeholder.com/110",width=110)

                st.success(
                    f"Result:\n\n**{st.session_state.last_run_gesture}**"
                )

    st.markdown("---")

# TRIGGER LOGIC
if "pending_run" in st.session_state:
    g_name = st.session_state.pop("pending_run")
    st.session_state.animation_running = True
    
    # Force the canvas to wipe completely clean before injecting the new animation
    animation_canvas.empty()
    
    run_inference_animation(g_name)
    
    # ---> Fetch real data from the CSV instead of random numbers <---
    # 1. Filter the master database for the exact combination selected
    match = master_db[
        (master_db["Dataset"] == dataset_choice) & 
        (master_db["Model"] == model_choice) & 
        (master_db["Input Gesture"] == g_name)
    ]
    
    # 2. Extract the values (with a safety fallback just in case a row is missing in the CSV)
    if not match.empty:
        real_latency = match["Latency (ms)"].values[0]
        real_ram = match["RAM (KB)"].values[0]
        real_energy = match["Energy (mJ)"].values[0]
        predicted_g = match["Predicted Gesture"].values[0]
    else:
        # Fallback if Excel data is missing for this specific combination
        real_latency, real_ram, real_energy, predicted_g = 0.0, 0.0, 0.0, "Not Found in CSV"

    # 3. Create the new row with your real hardware metrics
    new_row = pd.DataFrame([{
        "Model": model_choice,
        "Input Gesture": g_name,
        "Predicted Gesture": predicted_g,
        "Latency (ms)": real_latency,
        "RAM (KB)": real_ram,
        "Energy (mJ)": real_energy
    }])
    
    # Add the new row to the table and rerun
    st.session_state.results_tables[dataset_choice] = pd.concat([st.session_state.results_tables[dataset_choice], new_row], ignore_index=True)
    st.session_state.animation_running = False
    st.rerun()
    
# Results 
st.subheader(f"📊 {dataset_choice} Results")

# Retrieve and copy the current results DataFrame so we don't alter the raw stored data
df = st.session_state.results_tables[dataset_choice].copy()

# Format the numeric columns to clean up the decimal trails
if not df.empty:
    df["Latency (ms)"] = df["Latency (ms)"].map("{:.5f}".format)
    df["RAM (KB)"] = df["RAM (KB)"].map("{:.5f}".format)
    df["Energy (mJ)"] = df["Energy (mJ)"].map("{:.5f}".format)

# Convert the DataFrame to an HTML table
html_table = df.to_html(index=False, classes="center-text-table")

# Inject custom CSS and the HTML table directly into the app
st.markdown(f"""
<style>
.center-text-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}}
/* Force absolute centering for headers and cells */
.center-text-table th, .center-text-table td {{
    text-align: center !important;
    padding: 12px;
    border-bottom: 1px solid #ddd;
}}
/* Light background for the header to match Streamlit's default aesthetic */
.center-text-table th {{
    background-color: #f0f2f6;
    color: #31333F;
}}
</style>

{html_table}
""", unsafe_allow_html=True)
