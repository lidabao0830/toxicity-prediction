import os
import sys
import time
import uuid
import pathlib
import inspect
import contextlib
import tempfile

import pandas as pd
import streamlit as st


MATPLOTLIB_AVAILABLE = False
MATPLOTLIB_IMPORT_ERROR = ""

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = str(e)


from toxicity_platform import ToxicityPredictionPlatform


# =========================================================
# 你提供给用户使用的默认模型路径
# 后台固定平台模型，不在前端让用户选择或上传
# =========================================================
PLATFORM_DEFAULT_MODEL_PATH = r"Best-Train-Model.pth"


# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="化合物毒性预测系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 样式
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

:root {
    --primary: #1d4f91;
    --primary-dark: #143b6c;
    --primary-light: #eaf2fb;
    --accent: #2f80ed;
    --bg: #f4f8fc;
    --bg-soft: #f8fbff;
    --card: #ffffff;
    --border: #d6e1ee;
    --border-strong: #bfd0e3;
    --text: #17324d;
    --text-light: #5f738a;
    --danger: #dc2626;
    --success: #16a34a;
    --warning: #d97706;
    --shadow-sm: 0 4px 10px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 10px 26px rgba(15, 23, 42, 0.08);
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 18px;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif !important;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(47,128,237,0.08) 0%, rgba(47,128,237,0.02) 18%, transparent 34%),
        linear-gradient(180deg, #f8fbff 0%, #f4f8fc 55%, #eef4fb 100%);
}

.block-container {
    max-width: 1480px;
    padding-top: 2.25rem !important;
    padding-bottom: 1.6rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

div[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem !important;
    padding-left: 1.15rem !important;
    padding-right: 1.15rem !important;
}

.page-header {
    background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px 24px 18px 24px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #153554;
    line-height: 1.38;
    letter-spacing: 0.2px;
    margin: 0 0 8px 0;
    padding-top: 2px;
    word-break: break-word;
}

.sub-title {
    font-size: 14px;
    color: var(--text-light);
    line-height: 1.85;
    margin: 0;
    word-break: break-word;
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #173b63;
    margin: 6px 0 14px 0;
    line-height: 1.45;
    letter-spacing: 0.1px;
}

.small-title {
    font-size: 20px;
    font-weight: 800;
    color: #173b63;
    margin-bottom: 12px;
    line-height: 1.45;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
}

.hero-card {
    background: linear-gradient(135deg, #163a67 0%, #2c6cb0 100%);
    border-radius: 18px;
    padding: 30px 30px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 14px 34px rgba(21, 58, 102, 0.18);
    border: 1px solid rgba(255,255,255,0.08);
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 10px;
    line-height: 1.45;
    letter-spacing: 0.2px;
}

.hero-sub {
    font-size: 14px;
    line-height: 1.9;
    color: rgba(255,255,255,0.96);
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 100px;
    box-shadow: var(--shadow-sm);
}

.metric-label {
    font-size: 13px;
    color: var(--text-light);
    margin-bottom: 6px;
    font-weight: 600;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #17324d;
    line-height: 1.35;
}

.quick-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 156px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.quick-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--border-strong);
}

.quick-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.quick-title {
    font-size: 18px;
    font-weight: 800;
    color: #173b63;
    line-height: 1.45;
}

.sidebar-title {
    font-size: 20px;
    font-weight: 800;
    color: #173b63;
    margin-bottom: 8px;
    line-height: 1.45;
}

.path-label {
    font-size: 14px;
    font-weight: 700;
    color: #223a57;
    margin: 2px 0 8px 0;
    line-height: 1.5;
}

.stTextInput > div > div > input,
.stNumberInput input,
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.2px solid var(--border) !important;
    background: #ffffff !important;
    color: var(--text) !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #6ea3da !important;
    box-shadow: 0 0 0 3px rgba(47,128,237,0.12) !important;
}

.stTextInput > div > div > input,
.stNumberInput input {
    height: 46px !important;
    min-height: 46px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
}

.stTextInput {
    width: 100% !important;
}
.stTextInput > div {
    width: 100% !important;
}
.stTextInput > div > div {
    width: 100% !important;
}

.stTextArea textarea {
    border-radius: 10px !important;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    min-height: 44px !important;
    border: 1.2px solid var(--border) !important;
    background: #ffffff !important;
    color: #173b63 !important;
    font-weight: 700 !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
}

div[data-testid="stButton"] > button:hover {
    border-color: #8fb1d4 !important;
    color: #123b67 !important;
    background: #f8fbff !important;
}

div[data-testid="stButton"] > button:focus {
    box-shadow: 0 0 0 3px rgba(47,128,237,0.12) !important;
}

.primary-btn button {
    background: linear-gradient(135deg, #245b8f 0%, #2f80ed 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 18px rgba(47,128,237,0.22) !important;
}

.primary-btn button:hover {
    background: linear-gradient(135deg, #214f7c 0%, #2a73d6 100%) !important;
    color: white !important;
}

.icon-btn button {
    width: 100% !important;
    min-width: 56px !important;
    height: 46px !important;
    min-height: 46px !important;
    border-radius: 10px !important;
    padding: 0 !important;
    font-size: 18px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    margin-bottom: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    height: 46px;
    font-weight: 700;
    background: #f5f8fc;
    border: 1px solid var(--border);
    border-bottom: none;
    padding-left: 18px;
    padding-right: 18px;
}

.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--primary-dark) !important;
}

.log-panel {
    background: #fbfdff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
}

.path-preview {
    margin-top: 8px;
    background: #f8fbff;
    border: 1px dashed var(--border-strong);
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 12px;
    line-height: 1.7;
    color: #5d7086;
    word-break: break-all;
}

.path-preview strong {
    color: #173b63;
}

.result-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    margin-top: 10px;
    box-shadow: var(--shadow-sm);
}

div[data-testid="stRadio"] label {
    font-weight: 600 !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--text-light) !important;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 18px 0;
}

div[data-baseweb="notification"] {
    border-radius: 12px !important;
}

.top-spacer-fix {
    height: 2px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Session 初始化
# =========================================================
def init_session():
    defaults = {
        "platform": None,
        "quick_platform": None,
        "quick_model_loaded": False,
        "quick_model_error": "",

        "graphs": None,
        "logs": [],
        "log_images": [],
        "current_page": "首页",
        "current_model_path": "",
        "current_data_path": "",
        "data_loaded_count": 0,
        "train_plot_files": [],
        "train_single_result": None,
        "load_single_result": None,
        "quick_single_result": None,
        "cv_tracker": None,

        # 训练
        "train_save_dir": "./trained_models",
        "train_csv_path": "",
        "train_smiles_col": "SMILES",
        "train_label_col": "Toxicity_Label",
        "training_mode": "cv",
        "n_folds": 5,
        "test_size": 0.2,
        "random_state": 42,
        "n_epochs": 200,
        "batch_size": 64,
        "num_layers": 2,
        "hidden_dim": 128,
        "dropout": 0.3,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,

        # 损失函数
        "loss_type": "focal",
        "focal_alpha": 0.85,
        "focal_gamma": 1.7,

        "train_model_name": "",
        "train_single_smiles": "",
        "train_batch_input": "",
        "train_batch_output_dir": os.getcwd(),
        "train_batch_output_filename": "prediction_results.csv",
        "train_batch_smiles_col": "SMILES",

        # 加载模型预测
        "load_model_path": "",
        "load_single_smiles": "",
        "load_batch_input": "",
        "load_batch_output_dir": os.getcwd(),
        "load_batch_output_filename": "prediction_results.csv",
        "load_batch_smiles_col": "SMILES",

        # 免训练预测（后台固定平台模型）
        "quick_single_smiles": "",
        "quick_batch_input": "",
        "quick_batch_output_dir": os.getcwd(),
        "quick_batch_output_filename": "prediction_results.csv",
        "quick_batch_smiles_col": "SMILES",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# =========================================================
# 日志写入器
# =========================================================
class StreamlitLogWriter:
    def __init__(self, placeholder=None, also_write_to_terminal=True):
        self.placeholder = placeholder
        self.also_write_to_terminal = also_write_to_terminal
        self._buffer = ""
        self._terminal = sys.__stdout__

    def write(self, message):
        if not message:
            return

        if self.also_write_to_terminal:
            self._terminal.write(message)
            self._terminal.flush()

        self._buffer += message

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                st.session_state.logs.append(line)
                self._refresh()

    def flush(self):
        if self._buffer.strip():
            st.session_state.logs.append(self._buffer.strip())
            self._buffer = ""
            self._refresh()

        if self.also_write_to_terminal:
            self._terminal.flush()

    def _refresh(self):
        if self.placeholder is not None:
            txt = "\n".join(st.session_state.logs[-500:])
            self.placeholder.text_area(
                "输出日志",
                value=txt,
                height=420,
                key=f"log_live_{len(st.session_state.logs)}"
            )


# =========================================================
# 工具函数
# =========================================================
def append_log(msg):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {msg}")


def clear_logs():
    st.session_state.logs = []
    st.session_state.log_images = []
    st.session_state.train_plot_files = []


def ensure_platform(save_dir="./trained_models"):
    os.makedirs(save_dir, exist_ok=True)
    if st.session_state.platform is None:
        st.session_state.platform = ToxicityPredictionPlatform(model_save_dir=save_dir)
    else:
        st.session_state.platform.model_save_dir = save_dir


def ensure_quick_platform(save_dir="./trained_models"):
    os.makedirs(save_dir, exist_ok=True)
    if st.session_state.quick_platform is None:
        st.session_state.quick_platform = ToxicityPredictionPlatform(model_save_dir=save_dir)
    else:
        st.session_state.quick_platform.model_save_dir = save_dir


def page_jump(target_page):
    st.session_state.current_page = target_page
    st.rerun()


def choose_file_dialog(title="选择文件", filetypes=(("All files", "*.*"),)):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path
    except Exception as e:
        append_log(f"⚠️ 无法打开文件选择窗口：{e}")
        return ""


def choose_directory_dialog(title="选择目录"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path
    except Exception as e:
        append_log(f"⚠️ 无法打开目录选择窗口：{e}")
        return ""


def sync_text_value(real_key):
    widget_key = f"{real_key}__widget"
    st.session_state[real_key] = st.session_state.get(widget_key, "")


def path_input(label, key, is_dir=False, filetypes=(("All files", "*.*"),)):
    widget_key = f"{key}__widget"
    pending_key = f"{key}__pending"

    if pending_key in st.session_state:
        pending_value = st.session_state[pending_key]
        st.session_state[key] = pending_value
        st.session_state[widget_key] = pending_value
        del st.session_state[pending_key]
    elif widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, "")

    st.markdown(f'<div class="path-label">{label}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([16, 2], vertical_alignment="bottom")

    with c1:
        st.text_input(
            label,
            key=widget_key,
            label_visibility="collapsed",
            placeholder=f"请输入{label}路径",
            on_change=sync_text_value,
            args=(key,)
        )

    with c2:
        st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
        if st.button("📁" if is_dir else "📂", key=f"{key}_picker", help=f"浏览{label}"):
            selected = choose_directory_dialog(label) if is_dir else choose_file_dialog(label, filetypes=filetypes)
            if selected:
                st.session_state[pending_key] = selected
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    current_value = st.session_state.get(widget_key, "") or st.session_state.get(key, "")
    if current_value:
        st.markdown(
            f'<div class="path-preview"><strong>当前路径：</strong>{current_value}</div>',
            unsafe_allow_html=True
        )


def render_prediction_result(result, title="预测结果"):
    if not result:
        st.warning("未获得预测结果")
        return

    pred = str(result.get("prediction", ""))
    prob = float(result.get("probability", 0))
    conf = float(result.get("confidence", 0))
    smiles = result.get("smiles", "")

    with st.container():
        st.markdown(f"### {title}")
        st.markdown("**规范化 SMILES**")
        st.code(smiles, language=None)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**预测结果**")
            if "毒" in pred or "toxic" in pred.lower():
                st.error(pred)
            else:
                st.success(pred)

        with c2:
            st.markdown("**有毒概率**")
            st.info(f"{prob:.4f}")

        with c3:
            st.markdown("**预测置信度**")
            st.info(f"{conf:.4f}")

        st.progress(min(max(prob, 0.0), 1.0), text=f"有毒概率：{prob:.4f}")


def render_log_view():
    st.markdown('<div class="small-title">输出日志</div>', unsafe_allow_html=True)
    log_text = "\n".join(st.session_state.logs) if st.session_state.logs else "暂无日志"
    st.text_area("输出日志", value=log_text, height=500, key=f"logs_full_{len(st.session_state.logs)}")

    if st.session_state.log_images:
        st.markdown("#### 日志中的指标图像")
        cols = st.columns(2)
        for idx, img in enumerate(st.session_state.log_images):
            with cols[idx % 2]:
                if os.path.exists(img):
                    st.image(img, caption=os.path.basename(img), use_container_width=True)


def train_supports_progress_callback(platform_obj):
    try:
        sig = inspect.signature(platform_obj.train)
        return "progress_callback" in sig.parameters
    except Exception:
        return False


def train_supports_loss_params(platform_obj):
    try:
        sig = inspect.signature(platform_obj.train)
        return all(p in sig.parameters for p in ["loss_type", "focal_alpha", "focal_gamma"])
    except Exception:
        return False


def ensure_log_image_dir():
    img_dir = os.path.join(tempfile.gettempdir(), "toxicity_streamlit_logs")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def render_training_config_block(platform_obj, title="当前模型训练参数"):
    if platform_obj is None:
        st.info("当前没有可显示的模型配置")
        return

    cfg = getattr(platform_obj, "training_config", None) or {}
    if not cfg:
        st.info("该模型文件中未检测到训练参数信息")
        return

    st.markdown(f"#### {title}")
    lines = [f"- **{k}**: {v}" for k, v in cfg.items()]
    st.markdown("\n".join(lines))


def load_model_to_platform(model_path, save_dir="./trained_models"):
    ensure_platform(save_dir)
    log_writer = StreamlitLogWriter(also_write_to_terminal=True)
    with st.spinner("正在加载模型..."):
        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            st.session_state.platform.load_model(model_path)
    log_writer.flush()
    st.session_state.current_model_path = model_path
    append_log(f"✅ 模型加载成功：{model_path}")


def load_model_to_quick_platform(model_path, save_dir="./trained_models"):
    ensure_quick_platform(save_dir)
    log_writer = StreamlitLogWriter(also_write_to_terminal=True)
    with st.spinner("正在自动加载平台模型..."):
        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            st.session_state.quick_platform.load_model(model_path)
    log_writer.flush()
    append_log(f"✅ 免训练平台模型加载成功：{model_path}")


def auto_load_quick_platform_model():
    """
    自动加载后台固定平台模型。
    使用独立 quick_platform，不影响“加载模型预测”页面中手动加载的模型。
    """
    ensure_quick_platform("./trained_models")

    if st.session_state.quick_model_loaded and st.session_state.quick_platform is not None \
       and getattr(st.session_state.quick_platform, "model", None) is not None:
        return True

    if not os.path.exists(PLATFORM_DEFAULT_MODEL_PATH):
        st.session_state.quick_model_loaded = False
        st.session_state.quick_model_error = f"后台平台模型不存在：{PLATFORM_DEFAULT_MODEL_PATH}"
        append_log(f"❌ {st.session_state.quick_model_error}")
        return False

    try:
        load_model_to_quick_platform(PLATFORM_DEFAULT_MODEL_PATH, "./trained_models")
        st.session_state.quick_model_loaded = True
        st.session_state.quick_model_error = ""
        return True
    except Exception as e:
        st.session_state.quick_model_loaded = False
        st.session_state.quick_model_error = f"后台平台模型加载失败：{e}"
        append_log(f"❌ {st.session_state.quick_model_error}")
        return False


def fallback_predict_batch_to_csv(platform_obj, input_csv, output_csv, smiles_col="SMILES"):
    data = pd.read_csv(input_csv)
    if smiles_col not in data.columns:
        raise ValueError(f"列 '{smiles_col}' 不存在，可用列：{list(data.columns)}")

    predictions = []
    probabilities = []
    confidences = []

    append_log(f"📂 开始批量预测：{input_csv}")
    for idx, row in data.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or str(smiles).strip() == "":
            predictions.append("无效")
            probabilities.append(None)
            confidences.append(None)
        else:
            result = platform_obj.predict_single(str(smiles))
            if result:
                predictions.append(result["prediction"])
                probabilities.append(result["probability"])
                confidences.append(result["confidence"])
            else:
                predictions.append("无效")
                probabilities.append(None)
                confidences.append(None)

        if (idx + 1) % 100 == 0:
            append_log(f"   已处理：{idx + 1}/{len(data)}")

    data["Toxicity_Prediction"] = predictions
    data["Toxicity_Probability"] = probabilities
    data["Prediction_Confidence"] = confidences

    pathlib.Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False, encoding="utf-8-sig")
    append_log(f"✅ 批量预测完成，结果已保存：{output_csv}")
    return data


def platform_batch_predict(platform_obj, input_csv, output_csv, smiles_col="SMILES"):
    if hasattr(platform_obj, "predict_batch_from_csv") and callable(getattr(platform_obj, "predict_batch_from_csv")):
        return platform_obj.predict_batch_from_csv(input_csv, output_csv, smiles_col=smiles_col)
    return fallback_predict_batch_to_csv(platform_obj, input_csv, output_csv, smiles_col=smiles_col)


@contextlib.contextmanager
def capture_matplotlib_to_logs():
    original_show = plt.show

    def patched_show(*args, **kwargs):
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return
        save_dir = ensure_log_image_dir()
        for num in fig_nums:
            fig = plt.figure(num)
            filename = f"train_plot_{int(time.time())}_{uuid.uuid4().hex[:8]}_{num}.png"
            path = os.path.join(save_dir, filename)
            try:
                fig.savefig(path, dpi=160, bbox_inches="tight")
                st.session_state.log_images.append(path)
                st.session_state.train_plot_files.append(path)
                append_log(f"🖼️ 已生成训练图像：{os.path.basename(path)}")
            except Exception as e:
                append_log(f"⚠️ 图像保存失败：{e}")
        plt.close("all")

    plt.show = patched_show
    try:
        yield
    finally:
        plt.show = original_show


# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">化合物毒性预测系统</div>', unsafe_allow_html=True)
    st.caption("Graph Neural Network Toxicity Workbench")
    page_options = ["首页", "训练模型", "加载模型预测", "日志中心"]
    selected_page = st.radio(
        "导航",
        page_options,
        index=page_options.index(st.session_state.current_page)
    )
    st.session_state.current_page = selected_page


# =========================================================
# 顶部
# =========================================================
st.markdown('<div class="top-spacer-fix"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="main-title">化合物毒性预测系统</div>
    <div class="sub-title">
        基于图卷积神经网络的毒性模型训练、模型加载、单体预测、批量预测与训练日志可视化平台
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">模型状态</div>
        <div class="metric-value">{"已加载" if st.session_state.platform is not None and getattr(st.session_state.platform, "model", None) is not None else "未加载"}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">数据状态</div>
        <div class="metric-value">{"已加载" if st.session_state.graphs is not None else "未加载"}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">当前样本数</div>
        <div class="metric-value">{st.session_state.data_loaded_count}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 首页
# =========================================================
if st.session_state.current_page == "首页":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">专业的化合物毒性预测工作台</div>
        <div class="hero-sub">
            支持模型训练、交叉验证、普通训练、模型保存/加载、单样本预测、批量预测，
            并把训练输出与指标图像统一汇总到日志中心。
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon">🧠</div>
            <div class="quick-title">训练模型</div>
            <div style="margin-top:8px;color:#64748b;font-size:13px;line-height:1.7;">
                配置训练参数，执行普通训练或交叉验证，并跟踪训练日志与图像。
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入训练模型", use_container_width=True, key="home_to_train"):
            page_jump("训练模型")

    with c2:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon">📦</div>
            <div class="quick-title">加载模型预测</div>
            <div style="margin-top:8px;color:#64748b;font-size:13px;line-height:1.7;">
                加载已有模型文件，或直接使用平台提供的预训练模型进行预测。
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入加载模型预测", use_container_width=True, key="home_to_load"):
            page_jump("加载模型预测")

    with c3:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon">📋</div>
            <div class="quick-title">日志中心</div>
            <div style="margin-top:8px;color:#64748b;font-size:13px;line-height:1.7;">
                统一查看训练输出、模型加载信息、批量预测过程以及训练图像。
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入日志中心", use_container_width=True, key="home_to_log"):
            page_jump("日志中心")


# =========================================================
# 训练模型
# =========================================================
elif st.session_state.current_page == "训练模型":
    st.markdown('<div class="section-title">训练模型</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["数据", "参数", "训练", "保存与预测"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">数据设置</div>', unsafe_allow_html=True)

        path_input("保存目录", "train_save_dir", is_dir=True)
        path_input("训练 CSV", "train_csv_path", is_dir=False, filetypes=(("CSV files", "*.csv"),))

        a1, a2 = st.columns(2)
        with a1:
            st.text_input("SMILES列", key="train_smiles_col")
        with a2:
            st.text_input("标签列", key="train_label_col")

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("加载数据", key="btn_load_train_data", use_container_width=True):
            try:
                csv_path = st.session_state.train_csv_path.strip()
                save_dir = st.session_state.train_save_dir.strip() or "./trained_models"

                if not csv_path or not os.path.exists(csv_path):
                    st.error("请提供有效的训练 CSV 路径")
                    append_log("❌ 训练数据加载失败：CSV 路径无效")
                else:
                    ensure_platform(save_dir)

                    log_writer = StreamlitLogWriter(also_write_to_terminal=True)
                    with st.spinner("正在加载数据..."):
                        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                            graphs = st.session_state.platform.load_and_preprocess_data(
                                csv_path,
                                smiles_col=st.session_state.train_smiles_col.strip() or "SMILES",
                                label_col=st.session_state.train_label_col.strip() or "Toxicity_Label"
                            )
                    log_writer.flush()

                    st.session_state.graphs = graphs
                    st.session_state.current_data_path = csv_path
                    st.session_state.data_loaded_count = len(graphs) if graphs is not None else 0

                    st.success(f"数据加载完成，共 {st.session_state.data_loaded_count} 个样本")
                    append_log(f"✅ 数据加载完成：{csv_path}")
            except Exception as e:
                st.error(f"数据加载失败：{e}")
                append_log(f"❌ 数据加载失败：{e}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">训练参数</div>', unsafe_allow_html=True)

        st.radio(
            "训练模式",
            ["cv", "standard"],
            key="training_mode",
            horizontal=True,
            format_func=lambda x: "交叉验证" if x == "cv" else "普通训练"
        )

        if st.session_state.training_mode == "cv":
            st.info("当前模式：交叉验证训练")
        else:
            st.info("当前模式：普通训练")

        with st.form("train_params_form", clear_on_submit=False):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)

            with r1c1:
                if st.session_state.training_mode == "cv":
                    n_fold_options = [2, 3, 4, 5, 6, 7, 8, 9, 10]
                    current_folds = int(st.session_state.n_folds)
                    if current_folds not in n_fold_options:
                        current_folds = 5
                    n_folds = st.selectbox("折数", n_fold_options, index=n_fold_options.index(current_folds))
                else:
                    st.selectbox(
                        "折数（普通训练下不使用）",
                        [int(st.session_state.n_folds)],
                        index=0,
                        disabled=True
                    )
                    n_folds = int(st.session_state.n_folds)

            with r1c2:
                test_size = st.number_input("验证比例", min_value=0.05, max_value=0.95,
                                            value=float(st.session_state.test_size), step=0.05)

            with r1c3:
                random_state = st.number_input("随机种子", value=int(st.session_state.random_state), step=1)

            with r1c4:
                n_epochs = st.number_input("Epochs", min_value=1, max_value=5000,
                                           value=int(st.session_state.n_epochs), step=1)

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)

            with r2c1:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=2048,
                                             value=int(st.session_state.batch_size), step=1)

            with r2c2:
                num_layers = st.number_input("GCN层数", min_value=1, max_value=10,
                                             value=int(st.session_state.num_layers), step=1)

            with r2c3:
                hidden_dim = st.number_input("隐藏维度", min_value=8, max_value=8192,
                                             value=int(st.session_state.hidden_dim), step=1)

            with r2c4:
                dropout = st.number_input("Dropout", min_value=0.0, max_value=1.0,
                                          value=float(st.session_state.dropout), step=0.05)

            r3c1, r3c2 = st.columns(2)
            with r3c1:
                learning_rate = st.number_input("学习率", min_value=0.000001, max_value=0.1,
                                                value=float(st.session_state.learning_rate), format="%.6f")
            with r3c2:
                weight_decay = st.number_input("权重衰减", min_value=0.0, max_value=0.1,
                                               value=float(st.session_state.weight_decay), format="%.6f")

            st.markdown("---")
            st.markdown("#### 损失函数设置")

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                loss_type = st.selectbox(
                    "损失函数",
                    ["focal", "bce"],
                    index=0 if st.session_state.loss_type == "focal" else 1,
                    format_func=lambda x: "Focal Loss" if x == "focal" else "BCEWithLogitsLoss"
                )
            with lc2:
                focal_alpha = st.number_input(
                    "Focal Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.focal_alpha),
                    step=0.01,
                    format="%.2f",
                    disabled=(loss_type != "focal")
                )
            with lc3:
                focal_gamma = st.number_input(
                    "Focal Gamma",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state.focal_gamma),
                    step=0.1,
                    format="%.2f",
                    disabled=(loss_type != "focal")
                )

            submitted = st.form_submit_button("应用参数", use_container_width=True)

            if submitted:
                st.session_state.n_folds = int(n_folds)
                st.session_state.test_size = float(test_size)
                st.session_state.random_state = int(random_state)
                st.session_state.n_epochs = int(n_epochs)
                st.session_state.batch_size = int(batch_size)
                st.session_state.num_layers = int(num_layers)
                st.session_state.hidden_dim = int(hidden_dim)
                st.session_state.dropout = float(dropout)
                st.session_state.learning_rate = float(learning_rate)
                st.session_state.weight_decay = float(weight_decay)
                st.session_state.loss_type = str(loss_type)
                st.session_state.focal_alpha = float(focal_alpha)
                st.session_state.focal_gamma = float(focal_gamma)

                st.success("训练参数已更新")
                append_log(
                    f"✅ 训练参数已更新（loss_type={st.session_state.loss_type}, "
                    f"alpha={st.session_state.focal_alpha}, gamma={st.session_state.focal_gamma}）"
                )

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        left, right = st.columns([1.18, 1], gap="large")

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="small-title">训练控制</div>', unsafe_allow_html=True)

            st.caption(f"当前训练模式：{'交叉验证' if st.session_state.training_mode == 'cv' else '普通训练'}")
            st.caption(f"当前损失函数：{'Focal Loss' if st.session_state.loss_type == 'focal' else 'BCEWithLogitsLoss'}")
            if st.session_state.loss_type == "focal":
                st.caption(f"Focal Loss 参数：alpha={st.session_state.focal_alpha}, gamma={st.session_state.focal_gamma}")

            progress_bar = st.progress(0, text="等待开始")
            train_status = st.empty()

            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("开始训练", key="btn_start_train", use_container_width=True):
                try:
                    if st.session_state.graphs is None or len(st.session_state.graphs) == 0:
                        st.warning("请先加载训练数据")
                        append_log("⚠️ 请先加载训练数据")
                    else:
                        save_dir = st.session_state.train_save_dir.strip() or "./trained_models"
                        ensure_platform(save_dir)

                        st.session_state.train_plot_files = []
                        st.session_state.log_images = []

                        progress_bar.progress(5, text="正在准备训练")
                        train_status.info("训练进行中，请稍候...")

                        live_log_placeholder = st.empty()
                        log_writer = StreamlitLogWriter(
                            placeholder=live_log_placeholder,
                            also_write_to_terminal=True
                        )

                        train_kwargs = dict(
                            graphs=st.session_state.graphs,
                            training_mode=st.session_state.training_mode,
                            n_folds=int(st.session_state.n_folds),
                            test_size=float(st.session_state.test_size),
                            random_state=int(st.session_state.random_state),
                            n_epochs=int(st.session_state.n_epochs),
                            batch_size=int(st.session_state.batch_size),
                            hidden_dim=int(st.session_state.hidden_dim),
                            dropout=float(st.session_state.dropout),
                            learning_rate=float(st.session_state.learning_rate),
                            weight_decay=float(st.session_state.weight_decay),
                            num_layers=int(st.session_state.num_layers),
                        )

                        if train_supports_loss_params(st.session_state.platform):
                            train_kwargs["loss_type"] = st.session_state.loss_type
                            train_kwargs["focal_alpha"] = float(st.session_state.focal_alpha)
                            train_kwargs["focal_gamma"] = float(st.session_state.focal_gamma)
                        else:
                            append_log("⚠️ 当前 toxicity_platform.py 未检测到 Focal Loss 参数接口，若需真正启用 Focal Loss，请同步修改 toxicity_platform.py")

                        if train_supports_progress_callback(st.session_state.platform):
                            def progress_callback(stage, **kwargs):
                                try:
                                    if stage in ["start_cv", "start_standard"]:
                                        progress_bar.progress(10, text="训练开始")
                                    elif stage == "fold_start":
                                        fold = kwargs.get("fold", 0)
                                        progress_bar.progress(18, text=f"开始第 {fold + 1} 组训练")
                                    elif stage == "epoch_end":
                                        epoch = kwargs.get("epoch", 0) + 1
                                        total_epochs = kwargs.get("n_epochs", st.session_state.n_epochs)
                                        percent = min(95, 20 + int(epoch / max(total_epochs, 1) * 70))
                                        progress_bar.progress(percent, text=f"训练中：Epoch {epoch}/{total_epochs}")
                                    elif stage in ["end_cv", "end_standard"]:
                                        progress_bar.progress(100, text="训练完成")
                                except Exception:
                                    pass

                            train_kwargs["progress_callback"] = progress_callback

                        with st.spinner("正在训练模型，请稍候..."):
                            with capture_matplotlib_to_logs():
                                with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                                    cv_tracker = st.session_state.platform.train(**train_kwargs)

                        log_writer.flush()
                        st.session_state.cv_tracker = cv_tracker

                        progress_bar.progress(100, text="训练完成")
                        train_status.success("训练完成")
                        append_log("✅ 训练完成")

                        live_log_placeholder.text_area(
                            "输出日志",
                            value="\n".join(st.session_state.logs[-600:]),
                            height=420,
                            key=f"train_log_done_{len(st.session_state.logs)}"
                        )

                        st.success("训练完成，日志与指标图像已同步到日志区")
                except Exception as e:
                    st.error(f"训练失败：{e}")
                    append_log(f"❌ 训练失败：{e}")

            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_log_view()
            st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">保存模型</div>', unsafe_allow_html=True)

        s1, s2 = st.columns([5, 1], vertical_alignment="bottom")
        with s1:
            st.text_input("模型名称", key="train_model_name")
        with s2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("保存", key="btn_save_model", use_container_width=True):
                try:
                    if st.session_state.platform is None or getattr(st.session_state.platform, "model", None) is None:
                        st.warning("当前没有可保存的模型")
                        append_log("⚠️ 当前没有可保存的模型")
                    else:
                        st.session_state.platform.save_model(st.session_state.train_model_name.strip() or None)
                        st.success("模型保存完成")
                        append_log("✅ 模型保存完成")
                except Exception as e:
                    st.error(f"保存模型失败：{e}")
                    append_log(f"❌ 保存模型失败：{e}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="small-title">单个预测</div>', unsafe_allow_html=True)

        p1, p2 = st.columns([5, 1], vertical_alignment="bottom")
        with p1:
            st.text_input("单个 SMILES", key="train_single_smiles")
        with p2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("预测", key="btn_train_single_predict", use_container_width=True):
                try:
                    if st.session_state.platform is None or getattr(st.session_state.platform, "model", None) is None:
                        st.warning("请先训练或加载模型")
                        st.session_state.train_single_result = None
                    elif not st.session_state.train_single_smiles.strip():
                        st.warning("请输入 SMILES")
                        st.session_state.train_single_result = None
                    else:
                        result = st.session_state.platform.predict_single(st.session_state.train_single_smiles.strip())
                        st.session_state.train_single_result = result
                        append_log("✅ 单个预测完成")
                except Exception as e:
                    st.error(f"单个预测失败：{e}")
                    append_log(f"❌ 单个预测失败：{e}")
                    st.session_state.train_single_result = None
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.train_single_result:
            render_prediction_result(st.session_state.train_single_result, "单个预测结果")

        st.markdown("---")
        st.markdown('<div class="small-title">批量预测</div>', unsafe_allow_html=True)

        path_input("输入 CSV", "train_batch_input", is_dir=False, filetypes=(("CSV files", "*.csv"),))
        path_input("输出目录", "train_batch_output_dir", is_dir=True)

        b1, b2 = st.columns(2)
        with b1:
            st.text_input("输出文件名", key="train_batch_output_filename")
        with b2:
            st.text_input("SMILES列", key="train_batch_smiles_col")

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("批量预测", key="btn_train_batch_predict", use_container_width=True):
            try:
                if st.session_state.platform is None or getattr(st.session_state.platform, "model", None) is None:
                    st.warning("请先训练或加载模型")
                else:
                    input_csv = st.session_state.train_batch_input.strip()
                    output_dir = st.session_state.train_batch_output_dir.strip() or os.getcwd()
                    output_name = st.session_state.train_batch_output_filename.strip() or "prediction_results.csv"

                    if not input_csv or not os.path.exists(input_csv):
                        st.error("请提供有效的输入 CSV")
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, output_name)

                        log_writer = StreamlitLogWriter(also_write_to_terminal=True)
                        with st.spinner("正在执行批量预测..."):
                            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                                platform_batch_predict(
                                    st.session_state.platform,
                                    input_csv,
                                    output_path,
                                    smiles_col=st.session_state.train_batch_smiles_col.strip() or "SMILES"
                                )
                        log_writer.flush()

                        st.success(f"批量预测完成，结果已保存至：{output_path}")
                        append_log(f"✅ 批量预测完成：{output_path}")
            except Exception as e:
                st.error(f"批量预测失败：{e}")
                append_log(f"❌ 批量预测失败：{e}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 加载模型预测
# =========================================================
elif st.session_state.current_page == "加载模型预测":
    st.markdown('<div class="section-title">加载模型预测</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["加载模型", "单个预测", "批量预测", "免训练预测（平台预训练模型）"])

    # ---------------- 加载模型 ----------------
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">加载模型文件</div>', unsafe_allow_html=True)

        path_input("模型文件", "load_model_path", is_dir=False,
                   filetypes=(("Model files", "*.pth"), ("All files", "*.*")))

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("加载模型", key="btn_load_model", use_container_width=True):
            try:
                model_path = st.session_state.load_model_path.strip()
                if not model_path or not os.path.exists(model_path):
                    st.error("请提供有效的模型文件路径")
                    append_log("❌ 模型加载失败：模型文件路径无效")
                else:
                    load_model_to_platform(model_path, "./trained_models")
                    st.success(f"模型加载成功：{os.path.basename(model_path)}")
            except Exception as e:
                st.error(f"模型加载失败：{e}")
                append_log(f"❌ 模型加载失败：{e}")

        render_training_config_block(st.session_state.platform, "已加载模型的训练参数")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 单个预测 ----------------
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">单个预测</div>', unsafe_allow_html=True)

        q1, q2 = st.columns([5, 1], vertical_alignment="bottom")
        with q1:
            st.text_input("单个 SMILES", key="load_single_smiles")
        with q2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("预测", key="btn_load_single_predict", use_container_width=True):
                try:
                    if st.session_state.platform is None or getattr(st.session_state.platform, "model", None) is None:
                        st.warning("请先加载模型")
                        st.session_state.load_single_result = None
                    elif not st.session_state.load_single_smiles.strip():
                        st.warning("请输入 SMILES")
                        st.session_state.load_single_result = None
                    else:
                        result = st.session_state.platform.predict_single(st.session_state.load_single_smiles.strip())
                        st.session_state.load_single_result = result
                        append_log("✅ 单个预测完成")
                except Exception as e:
                    st.error(f"单个预测失败：{e}")
                    append_log(f"❌ 单个预测失败：{e}")
                    st.session_state.load_single_result = None
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.load_single_result:
            render_prediction_result(st.session_state.load_single_result, "单个预测结果")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 批量预测 ----------------
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">批量预测</div>', unsafe_allow_html=True)

        path_input("输入 CSV", "load_batch_input", is_dir=False, filetypes=(("CSV files", "*.csv"),))
        path_input("输出目录", "load_batch_output_dir", is_dir=True)

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("输出文件名", key="load_batch_output_filename")
        with c2:
            st.text_input("SMILES列", key="load_batch_smiles_col")

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("批量预测", key="btn_load_batch_predict", use_container_width=True):
            try:
                if st.session_state.platform is None or getattr(st.session_state.platform, "model", None) is None:
                    st.warning("请先加载模型")
                else:
                    input_csv = st.session_state.load_batch_input.strip()
                    output_dir = st.session_state.load_batch_output_dir.strip() or os.getcwd()
                    output_name = st.session_state.load_batch_output_filename.strip() or "prediction_results.csv"

                    if not input_csv or not os.path.exists(input_csv):
                        st.error("请提供有效的输入 CSV")
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, output_name)

                        log_writer = StreamlitLogWriter(also_write_to_terminal=True)
                        with st.spinner("正在执行批量预测..."):
                            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                                platform_batch_predict(
                                    st.session_state.platform,
                                    input_csv,
                                    output_path,
                                    smiles_col=st.session_state.load_batch_smiles_col.strip() or "SMILES"
                                )
                        log_writer.flush()

                        st.success(f"批量预测完成，结果已保存至：{output_path}")
                        append_log(f"✅ 批量预测完成：{output_path}")
            except Exception as e:
                st.error(f"批量预测失败：{e}")
                append_log(f"❌ 批量预测失败：{e}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 免训练预测（后台固定平台模型） ----------------
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">免训练预测</div>', unsafe_allow_html=True)

        st.info(
            "此模块不需要训练，也不需要前端上传模型。\n"
            "系统会自动加载后台固定的预训练平台模型，然后你可以直接进行单个 SMILES 或 CSV 批量预测。"
        )

        model_ok = auto_load_quick_platform_model()

        status_col1, status_col2 = st.columns([1, 3])
        with status_col1:
            if model_ok:
                st.success("平台模型已自动加载")
            else:
                st.error("平台模型加载失败")
        with status_col2:
            st.caption(f"后台模型路径：{PLATFORM_DEFAULT_MODEL_PATH}")

        if not model_ok:
            st.error(st.session_state.quick_model_error)
        else:
            render_training_config_block(st.session_state.quick_platform, "当前平台预训练模型的训练参数")

        st.markdown("---")
        st.markdown("#### 单个 SMILES 预测")

        p1, p2 = st.columns([5, 1], vertical_alignment="bottom")
        with p1:
            st.text_input("输入单个 SMILES", key="quick_single_smiles")
        with p2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("开始预测", key="btn_quick_single_predict", use_container_width=True):
                try:
                    if not model_ok or st.session_state.quick_platform is None or getattr(st.session_state.quick_platform, "model", None) is None:
                        st.warning("后台平台模型尚未成功加载")
                        st.session_state.quick_single_result = None
                    elif not st.session_state.quick_single_smiles.strip():
                        st.warning("请输入 SMILES")
                        st.session_state.quick_single_result = None
                    else:
                        result = st.session_state.quick_platform.predict_single(st.session_state.quick_single_smiles.strip())
                        st.session_state.quick_single_result = result
                        append_log("✅ 免训练单个预测完成")
                except Exception as e:
                    st.error(f"单个预测失败：{e}")
                    append_log(f"❌ 免训练单个预测失败：{e}")
                    st.session_state.quick_single_result = None
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.quick_single_result:
            render_prediction_result(st.session_state.quick_single_result, "免训练单个预测结果")

        st.markdown("---")
        st.markdown("#### CSV 批量预测")

        path_input("输入 CSV", "quick_batch_input", is_dir=False, filetypes=(("CSV files", "*.csv"),))
        path_input("输出目录", "quick_batch_output_dir", is_dir=True)

        qb1, qb2 = st.columns(2)
        with qb1:
            st.text_input("输出文件名", key="quick_batch_output_filename")
        with qb2:
            st.text_input("SMILES列", key="quick_batch_smiles_col")

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("开始批量预测", key="btn_quick_batch_predict", use_container_width=True):
            try:
                if not model_ok or st.session_state.quick_platform is None or getattr(st.session_state.quick_platform, "model", None) is None:
                    st.warning("后台平台模型尚未成功加载")
                else:
                    input_csv = st.session_state.quick_batch_input.strip()
                    output_dir = st.session_state.quick_batch_output_dir.strip() or os.getcwd()
                    output_name = st.session_state.quick_batch_output_filename.strip() or "prediction_results.csv"

                    if not input_csv or not os.path.exists(input_csv):
                        st.error("请提供有效的输入 CSV")
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, output_name)

                        log_writer = StreamlitLogWriter(also_write_to_terminal=True)
                        with st.spinner("正在执行批量预测..."):
                            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                                platform_batch_predict(
                                    st.session_state.quick_platform,
                                    input_csv,
                                    output_path,
                                    smiles_col=st.session_state.quick_batch_smiles_col.strip() or "SMILES"
                                )
                        log_writer.flush()

                        st.success(f"批量预测完成，结果已保存至：{output_path}")
                        append_log(f"✅ 免训练批量预测完成：{output_path}")
            except Exception as e:
                st.error(f"批量预测失败：{e}")
                append_log(f"❌ 免训练批量预测失败：{e}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 日志中心
# =========================================================
elif st.session_state.current_page == "日志中心":
    st.markdown('<div class="section-title">日志中心</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    l1, l2 = st.columns([1, 6], vertical_alignment="bottom")
    with l1:
        if st.button("清空日志", key="btn_clear_logs", use_container_width=True):
            clear_logs()
            st.success("日志已清空")
    with l2:
        st.caption("这里会显示训练输出、模型加载输出、批量预测输出，以及训练过程生成的指标图像。")

    render_log_view()

    st.markdown('</div>', unsafe_allow_html=True)
