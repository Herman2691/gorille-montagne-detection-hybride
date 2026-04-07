"""
🦍 Système Expert — Détection de Gorilles de Montagne
Architecture hybride : YOLOv11n + Module de Validation Heuristique (MVH)
Auteur : KANDOLO Herman · IS–VNU
"""

import os
import json
import traceback
import re
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import requests

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="🦍 Système Expert — Gorilles de Montagne",
    page_icon="🦍",
    layout="wide",
)

# ─── Configuration du Modèle ─────────────────────────────────────────────────
MODEL_PATH = "model.pt"
FILE_ID    = "1ojLIeI5Qbq1w0bSnaQm4YwJOzkdd2bUy"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Fonctions de Téléchargement ──────────────────────────────────────────────
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
            if match:
                token = match.group(1)

        if token:
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            resp = session.get(url, stream=True, timeout=120)

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
        
        return os.path.exists(dest) and os.path.getsize(dest) > 10000
    except Exception:
        return False

@st.cache_resource(show_spinner="⏳ Chargement de l'IA...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        ok = _download_from_drive(FILE_ID, MODEL_PATH)
        if not ok and os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        return None

# ─── Système Expert (Logique métier) ──────────────────────────────────────────
class ExpertKnowledge:
    SEUIL_C1, SEUIL_C3 = 0.70, 0.65
    POIDS_C1, POIDS_C2, POIDS_C3 = 0.40, 0.20, 0.40
    THETA_CONF, THETA_VETO, GAMMA = 0.70, 0.20, 0.50
    ALPHA_IA, ALPHA_MVH = 0.60, 0.40

    @staticmethod
    def calculer_SE(C1, C2_flag, C3):
        C2 = 1.0 if C2_flag else 0.0
        return round(0.4*C1 + 0.2*C2 + 0.4*C3, 4)

    @staticmethod
    def valider_classification(dl_class, C1, C3, C2_flag, score_ia):
        S_E = ExpertKnowledge.calculer_SE(C1, C2_flag, C3)
        preuves = [
            {"nom": "C₁ (Dos Argenté)", "score": C1, "seuil": 0.70, "validee": C1 >= 0.70},
            {"nom": "C₂ (Pelage)", "score": 1.0 if C2_flag else 0.0, "seuil": ">500", "validee": C2_flag},
            {"nom": "C₃ (Morphologie)", "score": C3, "seuil": 0.65, "validee": C3 >= 0.65},
        ]

        if score_ia >= ExpertKnowledge.THETA_CONF:
            if S_E >= ExpertKnowledge.THETA_VETO:
                return dl_class, S_E, score_ia, "Zone 1", "✅ IA Confirmée par Expertise", preuves
            else:
                return "Autres_gorilles", S_E, score_ia, "Zone 2", "⚠️ Véto Expertise (Traits absents)", preuves
        else:
            decision = round(0.6*score_ia + 0.4*S_E, 4)
            final = "Gorille_Montagne" if decision >= 0.5 else "Autres_gorilles"
            return final, S_E, decision, "Zone 3", f"🔄 Fusion Hybride (Score: {decision})", preuves

# ─── Vision par Ordinateur (MVH) ─────────────────────────────────────────────
def extraire_caracteristiques(image_array, bbox):
    try:
        x1, y1, x2, y2 = bbox
        roi = image_array[y1:y2, x1:x2]
        if roi.size == 0: return 0.0, 0.0, False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        h = gray.shape[0]

        # C1 : Dos Argenté (20-60%)
        dorsal = gray[int(h*0.2):int(h*0.6), :]
        C1 = min(1.0, (np.mean(dorsal)/255)*0.4 + (np.sum(dorsal>150)/dorsal.size)*0.6) if dorsal.size > 0 else 0.0

        # C3 : Morphologie Faciale (0-40%)
        facial = gray[0:int(h*0.4), :]
        if facial.size > 0:
            edges = cv2.Canny(facial, 50, 150)
            C3 = min(1.0, (np.sum(edges>0)/edges.size)*2.0 + (np.sum(facial<100)/facial.size)*0.5)
        else: C3 = 0.0

        # C2 : Texture
        C2_flag = cv2.Laplacian(gray, cv2.CV_64F).var() > 500
        return float(C1), float(C3), bool(C2_flag)
    except: return 0.0, 0.0, False

def dessiner(img, det, final_class, zone):
    x1, y1, x2, y2 = det["bbox"]
    color = (50, 205, 50) if final_class == "Gorille_Montagne" else (255, 165, 0)
    if zone == "Zone 2": color = (255, 60, 60)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, f"{final_class} ({zone})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# ─── Pipeline Principal ──────────────────────────────────────────────────────
def run_inference(image_pil, model):
    img_arr = np.array(image_pil.convert("RGB"))
    # Resize pour performance
    h, w = img_arr.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img_arr = cv2.resize(img_arr, (int(w*scale), int(h*scale)))

    res = model(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR), conf=0.25, verbose=False)
    
    if res and len(res[0].boxes) > 0:
        box = res[0].boxes[0]
        b = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        dl_name = "Gorille_Montagne" if cls_id == 1 else "Autres_gorilles"

        C1, C3, C2_f = extraire_caracteristiques(img_arr, b)
        det = {"DL": dl_name, "conf": conf, "C1": C1, "C3": C3, "C2_f": C2_f, "bbox": b}
        
        final, S_E, dec, zone, expl, preuv = ExpertKnowledge.valider_classification(dl_name, C1, C3, C2_f, conf)
        img_ann = dessiner(img_arr.copy(), det, final, zone)
        
        return img_ann, det, (final, S_E, dec, zone, expl, preuv)
    return None, None, None

# ─── Interface utilisateur ────────────────────────────────────────────────────
def main():
    # Hack pour éviter les erreurs de "removeChild" dues à la traduction auto
    st.markdown('<div class="notranslate">', unsafe_allow_html=True)
    
    st.title("🦍 Système Expert Gorilles — Architecture Hybride")
    st.caption("Déploiement IS-VNU | YOLOv11 + MVH")

    model = load_model()
    if not model:
        st.error("Impossible de charger le modèle .pt")
        return

    up = st.file_uploader("Image du spécimen", type=["jpg", "png", "jpeg", "webp"])
    
    if up:
        img_pil = Image.open(up)
        
        # On crée des conteneurs fixes pour stabiliser le DOM
        main_container = st.container()
        
        with main_container:
            col1, col2 = st.columns(2)
            with col1: st.image(img_pil, caption="Original", use_container_width=True)
            
            with st.spinner("Analyse hybride..."):
                img_res, det, expert = run_inference(img_pil, model)

            if img_res is not None:
                final_name, S_E, decision, zone, expl, preuv = expert
                
                with col2: st.image(img_res, caption="Analyse MVH", use_container_width=True)

                st.divider()
                # Affichage des métriques dans une zone stable
                m1, m2, m3 = st.columns(3)
                m1.metric("Zone Décisionnelle", zone)
                m2.metric("Score Expert (S_E)", f"{S_E:.3f}")
                m3.metric("Classe Finale", final_name)

                with st.expander("🔍 Détails du raisonnement"):
                    st.write(f"**Logique :** {expl}")
                    st.table(preuv)
            else:
                st.warning("Aucun gorille détecté sur cette image.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
