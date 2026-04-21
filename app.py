"""
🎨 VAE & Diffusion Model — Interactive Demo
=============================================
Streamlit application demonstrating Variational Autoencoders and 
Denoising Diffusion Probabilistic Models (DDPM).

TU-Varna | ИММО Project
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import time

from models.vae import VAE, train_vae_on_image, get_latent_interpolation, get_latent_space_samples
from models.diffusion import DiffusionModel, get_noise_schedule_visualization
from utils import preprocess_image, tensor_to_pil, numpy_to_pil

# ──────────────────────────── Page Configuration ────────────────────────────

st.set_page_config(
    page_title="VAE & Diffusion Demo | ТУ-Варна",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom Styles ────────────────────────────
st.markdown("""
<style>
    /* Clean overrides for a sharper look */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #8B5CF6;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────── Hero Header ───────────────────────────────────

st.title("🧬 VAE & Diffusion Models")
st.markdown("##### Интерактивна демонстрация на Variational Autoencoder (VAE) и Denoising Diffusion Probabilistic Model (DDPM)")
st.caption("Качете снимка и наблюдавайте как генеративните модели се обучават в реално време.")
st.divider()

# ──────────────────────────── Sidebar ───────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Настройки")
    
    st.subheader("📤 Качване на изображение")
    uploaded_file = st.file_uploader(
        "Изберете снимка (JPG, PNG)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    st.divider()
    st.subheader("🖼️ Общи")
    image_size = st.select_slider(
        "Резолюция за обучение (px)",
        options=[32, 64, 128],
        value=64,
        help="По-голяма резолюция = по-добро качество, но по-бавно обучение."
    )

    st.divider()
    st.subheader("🔮 VAE Параметри")
    vae_latent_dim = st.slider("Латентна размерност", 16, 256, 64, 16)
    vae_epochs = st.slider("Епохи (VAE)", 50, 800, 200, 50)
    vae_kl_weight = st.slider("KL тегло", 0.01, 2.0, 0.5, 0.01)
    vae_lr = st.selectbox("Learning Rate (VAE)", [0.0001, 0.0005, 0.001, 0.002, 0.005], index=2)

    st.divider()
    st.subheader("🌊 Diffusion Параметри")
    diff_timesteps = st.select_slider("Timesteps (T)", options=[50, 100, 200, 500, 1000], value=200)
    diff_epochs = st.slider("Епохи (Diffusion)", 100, 1500, 500, 100)
    diff_lr = st.selectbox("Learning Rate (Diffusion)", [0.0001, 0.0005, 0.001, 0.002], index=2)

    st.divider()
    st.caption("🎓 ТУ-Варна · ИММО Проект")

# ──────────────────────────── Helper Functions ──────────────────────────────

def create_plotly_loss_chart(history, title, y_keys, y_labels, colors):
    """Create a styled Plotly chart for loss visualization."""
    fig = go.Figure()
    for key, label, color in zip(y_keys, y_labels, colors):
        epochs = [h["epoch"] for h in history]
        values = [h[key] for h in history]
        fig.add_trace(go.Scatter(
            x=epochs, y=values,
            mode="lines",
            name=label,
            line=dict(color=color, width=3),
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Епоха", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title="Loss", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
    )
    return fig

def display_image_row(images, captions, num_cols=4):
    """Display a row of images with captions in columns."""
    cols = st.columns(num_cols)
    for i, (img, cap) in enumerate(zip(images, captions)):
        with cols[i]:
            st.image(img, caption=cap, width="stretch")


# ──────────────────────────── Main Content ─────────────────────────────────────

tab_vae, tab_diffusion = st.tabs(["🔮 VAE Модел", "🌊 Diffusion Модел"])

# ════════════════════════════ TAB: VAE ═══════════════════════════════════════

with tab_vae:
    st.subheader("Variational Autoencoder (VAE)")
    
    if uploaded_file is None:
        st.info("👆 Моля, качете изображение от страничния панел, за да започнете.")
    else:
        # Preprocess
        image_tensor, original_image, resized_image = preprocess_image(uploaded_file, image_size)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(resized_image, caption=f"Вход (Resized to {image_size}x{image_size})", width="stretch")
            
        with c2:
            st.write("#### Текущи параметри")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Латентна Dim", vae_latent_dim)
            m2.metric("Епохи", vae_epochs)
            m3.metric("KL Тегло", vae_kl_weight)
            m4.metric("Learning Rate", vae_lr)

        # Training
        if st.button("🚀 Стартирай VAE обучение", type="primary", use_container_width=True):
            st.divider()
            st.write("#### 🏋️ Обучение в прогрес...")

            prog_bar = st.progress(0)
            status_text = st.empty()

            def vae_progress(epoch, total, loss_dict):
                prog_bar.progress(epoch / total)
                status_text.markdown(f"**Епоха {epoch}/{total}** | Total: `{loss_dict['total_loss']:.2f}` | Recon: `{loss_dict['recon_loss']:.2f}` | KL: `{loss_dict['kl_loss']:.2f}`")

            start_t = time.time()
            model, history = train_vae_on_image(
                image_tensor, latent_dim=vae_latent_dim, num_epochs=vae_epochs,
                lr=vae_lr, kl_weight=vae_kl_weight, progress_callback=vae_progress
            )
            elapsed = time.time() - start_t
            prog_bar.progress(1.0)
            status_text.success(f"✅ Обучението завърши за {elapsed:.2f} секунди.")

            st.session_state["vae_model"] = model
            
            # Display results
            st.write("### 📉 Loss History")
            fig_loss = create_plotly_loss_chart(history, "VAE Loss Components", ["total_loss", "recon_loss", "kl_loss"], ["Total", "Reconstruction", "KL Div"], ["#EF4444", "#3B82F6", "#10B981"])
            st.plotly_chart(fig_loss, use_container_width=True)

            st.write("### 🖼️ Реконструкция")
            with torch.no_grad():
                device = next(model.parameters()).device
                reconstruction, mu, log_var, z = model(image_tensor.to(device))
                recon_img = tensor_to_pil(reconstruction)
            
            r1, r2 = st.columns(2)
            r1.image(resized_image, caption="Оригинал", width="stretch")
            r2.image(recon_img, caption="Реконструирано", width="stretch")
            
            st.write("### 🔄 Латентна интерполация")
            with torch.no_grad():
                z1, _, _ = model.encode(image_tensor.to(device))
                z2 = torch.randn_like(z1)
                interp_images = get_latent_interpolation(model, z1, z2, steps=8)
            interp_pil = [numpy_to_pil(img) for img in interp_images]
            interp_cap = [f"α = {a:.2f}" for a in np.linspace(0, 1, 8)]
            display_image_row(interp_pil, interp_cap, 8)
            
            st.write("### 🎲 Случайни проби от N(0, I)")
            samples = get_latent_space_samples(model, num_samples=8, latent_dim=vae_latent_dim)
            sample_pil = [numpy_to_pil(img) for img in samples]
            display_image_row(sample_pil, [f"Sample {i+1}" for i in range(8)], 8)


# ════════════════════════════ TAB: Diffusion ═════════════════════════════════

with tab_diffusion:
    st.subheader("Denoising Diffusion Probabilistic Model (DDPM)")
    
    if uploaded_file is None:
        st.info("👆 Моля, качете изображение от страничния панел, за да започнете.")
    else:
        image_tensor, original_image, resized_image = preprocess_image(uploaded_file, image_size)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(resized_image, caption=f"Вход (Resized to {image_size}x{image_size})", width="stretch")
            
        with c2:
            st.write("#### Текущи параметри")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Timesteps T", diff_timesteps)
            m2.metric("Епохи", diff_epochs)
            m3.metric("Базова архитектура", "U-Net")
            m4.metric("Learning Rate", diff_lr)

        st.write("### 📸 Forward Process (Добавяне на шум)")
        with st.spinner("Генериране на forward процес..."):
            diff_model = DiffusionModel(num_timesteps=diff_timesteps, image_size=image_size, in_channels=3)
            forward_viz = diff_model.get_forward_process_visualization(image_tensor, num_steps=8)
        fw_img = [numpy_to_pil(i) for _, i in forward_viz]
        fw_cap = [f"t = {t}" for t, _ in forward_viz]
        display_image_row(fw_img, fw_cap, 8)

        # Training
        if st.button("🚀 Стартирай Diffusion обучение", type="primary", use_container_width=True):
            st.divider()
            st.write("#### 🏋️ Обучение в прогрес...")
            st.warning("⏳ Забележка: Diffusion моделите отнемат малко повече време за обучение.")

            prog_bar_d = st.progress(0)
            status_text_d = st.empty()

            def diff_progress(epoch, total, loss_dict):
                prog_bar_d.progress(epoch / total)
                status_text_d.markdown(f"**Епоха {epoch}/{total}** | MSE Loss: `{loss_dict['loss']:.6f}`")

            start_t = time.time()
            diff_model_inst = DiffusionModel(num_timesteps=diff_timesteps, image_size=image_size, in_channels=3)
            diff_history = diff_model_inst.train_on_image(
                image_tensor, num_epochs=diff_epochs, lr=diff_lr, progress_callback=diff_progress
            )
            elapsed = time.time() - start_t
            prog_bar_d.progress(1.0)
            status_text_d.success(f"✅ Обучението завърши за {elapsed:.2f} секунди.")
            
            st.write("### 📉 Loss History")
            fig_diff = create_plotly_loss_chart(diff_history, "Training Loss (Noise Prediction)", ["loss"], ["MSE Loss"], ["#06B6D4"])
            st.plotly_chart(fig_diff, use_container_width=True)
            
            st.write("### 🎨 Обратен процес (Генерация от чист шум)")
            st.write("Моделът стартира от гаусов шум и итеративно го премахва стъпка по стъпка.")
            
            prog_sample = st.progress(0)
            stat_sample = st.empty()
            
            def sample_prog(step, total):
                prog_sample.progress(step / total)
                stat_sample.markdown(f"**Стъпка {step}/{total}** (Denoising...)")
                
            generated, intermediates = diff_model_inst.sample(num_samples=1, progress_callback=sample_prog)
            prog_sample.progress(1.0)
            stat_sample.success("✅ Генерацията завърши успешно!")
            
            inter_img = [numpy_to_pil(i) for _, i in intermediates]
            inter_cap = [f"t = {t}" for t, _ in intermediates]
            
            # Select evenly spaced screenshots
            num_show = min(len(inter_img), 8)
            step = max(1, len(inter_img) // num_show)
            selected_img = inter_img[::step][:num_show]
            selected_cap = inter_cap[::step][:num_show]
            
            display_image_row(selected_img, selected_cap, len(selected_img))
            
            st.write("### 🔍 Финален резултат")
            f1, f2 = st.columns(2)
            f1.image(resized_image, caption="Оригинал", width="stretch")
            f2.image(numpy_to_pil(intermediates[-1][1]), caption="Генерирано", width="stretch")
