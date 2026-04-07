"""
🦍 Système Expert — Détection de Gorilles de Montagne
Architecture hybride : YOLOv11n + Module de Validation Heuristique (MVH)
Auteur : KANDOLO Herman · IS–VNU
"""

import os
import json
import re
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import requests

# ─── 1. CONFIGURATION DE LA PAGE (Anti-Error DOM) ───────────────────────────
st.set_page_config(
    page_title="🦍 Système Expert — Gorilles",
    page_icon="🦍",
    layout="wide",
)

# Configuration du Modèle (Lien Drive vérifié)
MODEL_PATH = "model.pt"
FILE_ID    = "1ojLIeI5Qbq1w0bSnaQm4YwJOzkdd2bUy"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 2. TÉLÉCHARGEMENT ROBUSTE DEPUIS DRIVE ──────────────────────────────────
def _download_from_drive(file_id: str, dest: str) -> bool:
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        resp = session.get(url, stream=True, timeout=30)
        token = None
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                token = val
                break
        if token is None:
            match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
            if match: token = match.group(1)

        if token:
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            resp = session.get(url, stream=True, timeout=120)

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
        return os.path.exists(dest) and os.path.getsize(dest) > 10000
    except:
        return False

@st.cache_resource(show_spinner="⏳ Chargement de l'IA (YOLOv11)...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        if not _download_from_drive(FILE_ID, MODEL_PATH):
            st.error("❌ Impossible de télécharger le modèle. Vérifiez les permissions Drive.")
            return None
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"❌ Erreur technique lors du chargement : {e}")
        return None

# ─── 3. SYSTÈME EXPERT & VISION (MVH) ───────────────────────────────────────
class ExpertKnowledge:
    # Paramètres conformes à votre recherche SIM
    THETA_CONF, THETA_VETO, GAMMA = 0.70, 0.20, 0.50
    ALPHA_IA, ALPHA_MVH = 0.60, 0.40

    @staticmethod
    def valider(dl_class, C1, C3, C2_f, score_ia):
        S_E = round(0.4*C1 + 0.2*(1.0 if C2_f else 0.0) + 0.4*C3, 4)
        preuves = [
            {"Critère": "C₁ (Dos Argenté)", "Valeur": f"{C1:.3f}", "Validé": "✅" if C1 >= 0.70 else "❌"},
            {"Critère": "C₂ (Densité Pelage)", "Valeur": "OK" if C2_f else "Faible", "Validé": "✅" if C2_f else "❌"},
            {"Critère": "C₃ (Morphologie Faciale)", "Valeur": f"{C3:.3f}", "Validé": "✅" if C3 >= 0.65 else "❌"},
        ]

        if score_ia >= ExpertKnowledge.THETA_CONF:
            if S_E >= ExpertKnowledge.THETA_VETO:
                return dl_class, S_E, score_ia, "Zone 1", "✅ IA Confirmée par Expertise", preuves
            return "Autres_gorilles", S_E, score_ia, "Zone 2", "⚠️ Véto MVH : Traits morphologiques absents", preuves
        
        decision = round(ExpertKnowledge.ALPHA_IA * score_ia + ExpertKnowledge.ALPHA_MVH * S_E, 4)
        final = "Gorille_Montagne" if decision >= ExpertKnowledge.GAMMA else "Autres_gorilles"
        return final, S_E, decision, "Zone 3", f"🔄 Fusion Hybride (Score: {decision})", preuves

def extract_mvh(img_arr, bbox):
    try:
        x1, y1, x2, y2 = bbox
        roi = img_arr[y1:y2, x1:x2]
        if roi.size == 0: return 0.0, 0.0, False
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        h = gray.shape[0]
        
        # C1 (20-60%) et C3 (0-40%)
        dorsal = gray[int(h*0.2):int(h*0.6), :]
        facial = gray[0:int(h*0.4), :]
        
        C1 = min(1.0, (np.mean(dorsal)/255)*0.4 + (np.sum(dorsal>150)/dorsal.size)*0.6) if dorsal.size > 0 else 0.0
        C3 = min(1.0, (np.sum(cv2.Canny(facial, 50, 150)>0)/facial.size)*2.0 + (np.sum(facial<100)/facial.size)*0.5) if facial.size > 0 else 0.0
        C2_f = cv2.Laplacian(gray, cv2.CV_64F).var() > 500
        
        return C1, C3, C2_f
    except: return 0.0, 0.0, False

# ─── 4. INTERFACE UTILISATEUR ────────────────────────────────────────────────
def main():
    # PROTECTION CRUCIALE : Empêche Google Traduction de casser le DOM
    st.markdown('<div class="notranslate">', unsafe_allow_html=True)
    
    st.title("🦍 Système Expert Gorilles — Architecture Hybride")
    st.markdown("> **YOLOv11 + Module de Validation Heuristique (MVH)**")

    model = load_model()
    if not model: return

    file = st.file_uploader("📷 Uploader une image de gorille", type=["jpg", "png", "jpeg", "webp"])
    
    if file:
        img_pil = Image.open(file).convert("RGB")
        
        # Structure de page stable
        ui_images = st.container()
        ui_results = st.container()

        with ui_images:
            col_in, col_out = st.columns(2)
            col_in.image(img_pil, caption="Image d'entrée", use_container_width=True)

        with st.spinner("🧠 Analyse experte en cours..."):
            img_arr = np.array(img_pil)
            # Inférence YOLO
            res = model(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR), conf=0.25, verbose=False)
            
            if res and len(res[0].boxes) > 0:
                box = res[0].boxes[0]
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                dl_name = "Gorille_Montagne" if int(box.cls[0].cpu().numpy()) == 1 else "Autres_gorilles"

                # Validation Système Expert
                C1, C3, C2_f = extract_mvh(img_arr, b)
                final, S_E, dec_val, zone, expl, preuv = ExpertKnowledge.valider(dl_name, C1, C3, C2_f, conf)
                
                # Annotation visuelle
                img_res = img_arr.copy()
                color = (50, 205, 50) if final == "Gorille_Montagne" else (255, 165, 0)
                if zone == "Zone 2": color = (255, 60, 60)
                cv2.rectangle(img_res, (b[0], b[1]), (b[2], b[3]), color, 3)
                cv2.putText(img_res, f"{final} ({zone})", (b[0], b[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                with col_out:
                    st.image(img_res, caption=f"Analyse : {final}", use_container_width=True)

                with ui_results:
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Zone de Décision", zone)
                    m2.metric("Score Expert (S_E)", f"{S_E:.3f}")
                    m3.metric("Verdict Final", final.replace("_", " "))

                    with st.expander("🔬 Détails du raisonnement heuristique"):
                        st.info(f"**Justification :** {expl}")
                        st.table(preuv)
            else:
                st.warning("⚠️ Aucun sujet détecté par l'IA sur cette image.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Auteur : KANDOLO Herman · Master 2 SIM · IFI/VNU · Hanoï 2026")

if __name__ == "__main__":
    main()
