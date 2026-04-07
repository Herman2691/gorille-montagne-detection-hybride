"""
🦍 Système Expert — Détection de Gorilles de Montagne
Architecture hybride : YOLOv11n + Module de Validation Heuristique (MVH)

Auteur : KANDOLO Herman · IS–VNU · hermankandolo2022@gmail.com
"""

import os
import json
import traceback
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="🦍 Système Expert — Gorilles de Montagne",
    page_icon="🦍",
    layout="wide",
)

# ─── Chargement du modèle depuis Google Drive ─────────────────────────────────
MODEL_PATH = "model.pt"
FILE_ID    = "1ojLIeI5Qbq1w0bSnaQm4YwJOzkdd2bUy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _download_from_drive(file_id: str, dest: str) -> bool:
    """
    Télécharge un fichier depuis Google Drive en contournant la page
    de confirmation antivirus (fichiers > 100 Mo).
    Stratégie : requests avec gestion du cookie de confirmation.
    """
    import requests

    session = requests.Session()

    # 1re requête — récupère le cookie de confirmation si besoin
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, stream=True, timeout=30)

    # Cherche le token de confirmation dans les cookies ou dans le HTML
    token = None
    for key, val in resp.cookies.items():
        if key.startswith("download_warning"):
            token = val
            break

    if token is None:
        # Google Drive peut maintenant renvoyer un formulaire HTML
        import re
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
        if match:
            token = match.group(1)

    if token:
        url = (
            f"https://drive.google.com/uc?export=download"
            f"&confirm={token}&id={file_id}"
        )
        resp = session.get(url, stream=True, timeout=120)

    # Vérifie que c'est bien un fichier binaire (pas une page HTML d'erreur)
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # Dernière tentative avec l'API export drive
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        resp = session.get(url, stream=True, timeout=120)
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            return False

    # Écriture sur disque par blocs
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    # Sanity-check : un .pt valide fait au moins quelques Ko
    return os.path.exists(dest) and os.path.getsize(dest) > 10_000


@st.cache_resource(show_spinner="⏳ Téléchargement et chargement du modèle…")
def load_model():
    """Télécharge le modèle depuis Google Drive si absent, puis le charge."""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Téléchargement du modèle depuis Google Drive…")
        ok = _download_from_drive(FILE_ID, MODEL_PATH)
        if not ok:
            # Nettoyage d'un éventuel fichier HTML corrompu
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ Téléchargement échoué. Vérifiez que :\n"
            "1. Le fichier Drive est partagé **« Tout le monde avec le lien »**\n"
            f"2. Le FILE_ID est correct : `{FILE_ID}`"
        )
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        # Warm-up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, conf=0.25, device=DEVICE, verbose=False)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PARAMÈTRES DU SYSTÈME EXPERT (conformes article v12)
# ═══════════════════════════════════════════════════════════════
class ExpertKnowledge:
    SEUIL_C1   = 0.70   # DCID — Dos Argenté
    SEUIL_C3   = 0.65   # FSGD — Morphologie Faciale
    POIDS_C1   = 0.40
    POIDS_C2   = 0.20
    POIDS_C3   = 0.40
    THETA_CONF = 0.70   # seuil confiance YOLO
    THETA_VETO = 0.20   # seuil véto MVH
    GAMMA      = 0.50   # seuil décisionnel Zone 3
    ALPHA_IA   = 0.60
    ALPHA_MVH  = 0.40

    @staticmethod
    def calculer_SE(C1, C2_flag, C3):
        C2 = 1.0 if C2_flag else 0.0
        return round(
            ExpertKnowledge.POIDS_C1 * C1
            + ExpertKnowledge.POIDS_C2 * C2
            + ExpertKnowledge.POIDS_C3 * C3,
            4,
        )

    @staticmethod
    def valider_classification(dl_classification, C1, C3, C2_flag, score_ia=0.0):
        if dl_classification == "Aucun Objet Détecté":
            return "Aucun Objet Détecté", 0.0, 0.0, "N/A", "Aucun gorille détecté.", []

        C2 = 1.0 if C2_flag else 0.0
        preuves = [
            {
                "nom": "C₁ — DCID (Dos Argenté)",
                "zone": "ROI 20–60 %",
                "score": C1,
                "seuil": ExpertKnowledge.SEUIL_C1,
                "poids": "40 %",
                "contribution": round(ExpertKnowledge.POIDS_C1 * C1, 4),
                "validee": C1 >= ExpertKnowledge.SEUIL_C1,
            },
            {
                "nom": "C₂ — FTDD (Densité Pelage)",
                "zone": "ROI entière",
                "score": C2,
                "seuil": "Var(ΔI) > 500",
                "poids": "20 %",
                "contribution": round(ExpertKnowledge.POIDS_C2 * C2, 4),
                "validee": C2_flag,
            },
            {
                "nom": "C₃ — FSGD (Morphologie Faciale)",
                "zone": "ROI 0–40 %",
                "score": C3,
                "seuil": ExpertKnowledge.SEUIL_C3,
                "poids": "40 %",
                "contribution": round(ExpertKnowledge.POIDS_C3 * C3, 4),
                "validee": C3 >= ExpertKnowledge.SEUIL_C3,
            },
        ]

        S_E        = ExpertKnowledge.calculer_SE(C1, C2_flag, C3)
        score_ia_n = max(0.0, min(1.0, score_ia))

        if score_ia_n >= ExpertKnowledge.THETA_CONF:
            if S_E >= ExpertKnowledge.THETA_VETO:
                # Zone 1
                return (
                    dl_classification, S_E, round(score_ia_n, 4), "Zone 1",
                    (f"✅ Zone 1 — YOLO certain + MVH confirme | "
                     f"ScoreIA={score_ia_n:.3f} ≥ {ExpertKnowledge.THETA_CONF} · "
                     f"S_E={S_E:.3f} ≥ θ_veto={ExpertKnowledge.THETA_VETO} → Décision YOLO acceptée"),
                    preuves,
                )
            else:
                # Zone 2 — Véto
                statut = (
                    "⚠️ Véto MVH" if dl_classification == "Gorille_Montagne"
                    else "✅ Véto MVH (confirme)"
                )
                return (
                    "Autres_gorilles", S_E, round(score_ia_n, 4), "Zone 2",
                    (f"{statut} | Zone 2 — ScoreIA={score_ia_n:.3f} ≥ {ExpertKnowledge.THETA_CONF} "
                     f"MAIS S_E={S_E:.3f} < θ_veto={ExpertKnowledge.THETA_VETO} → "
                     f"Absence traits morphologiques → Forcé : Autres_gorilles"),
                    preuves,
                )
        else:
            # Zone 3 — Fusion
            decision = round(
                ExpertKnowledge.ALPHA_IA * score_ia_n + ExpertKnowledge.ALPHA_MVH * S_E, 4
            )
            classif = "Gorille_Montagne" if decision >= ExpertKnowledge.GAMMA else "Autres_gorilles"
            if classif == dl_classification:
                statut = "Validé ✅"
            elif classif == "Gorille_Montagne":
                statut = "Corrigé ↑ ✅"
            else:
                statut = "Corrigé ↓"
            signe = "≥" if decision >= ExpertKnowledge.GAMMA else "<"
            return (
                classif, S_E, decision, "Zone 3",
                (f"{statut} | Zone 3 — YOLO incertain : ScoreIA={score_ia_n:.3f} < {ExpertKnowledge.THETA_CONF} → "
                 f"Fusion : {ExpertKnowledge.ALPHA_IA}×{score_ia_n:.3f} + {ExpertKnowledge.ALPHA_MVH}×{S_E:.3f} "
                 f"= {decision:.4f} {signe} Γ={ExpertKnowledge.GAMMA}"),
                preuves,
            )


# ═══════════════════════════════════════════════════════════════
# EXTRACTION DESCRIPTEURS MVH
# ═══════════════════════════════════════════════════════════════
def extraire_caracteristiques(image_array, bbox):
    try:
        x1, y1, x2, y2 = bbox
        h_img, w_img = image_array.shape[:2]
        x1 = max(0, min(x1, w_img - 1)); y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img)); y2 = max(y1 + 1, min(y2, h_img))

        roi = image_array[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return 0.0, 0.0, False
        if roi.dtype != np.uint8:
            roi = np.clip(roi, 0, 255).astype(np.uint8)
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        height, _ = roi_gray.shape

        # C1 — DCID (zone 20–60 %)
        d1 = max(1, int(height * 0.20)); d2 = max(d1 + 1, int(height * 0.60))
        dorsal = roi_gray[d1:d2, :]
        if dorsal.size == 0:
            C1 = 0.0
        else:
            L_mean  = float(np.mean(dorsal))
            sigma_L = float(np.std(dorsal))
            R_b     = float(np.sum(dorsal > 150)) / dorsal.size
            C1 = min(1.0, (L_mean / 255.0) * 0.4 + R_b * 0.4 + (sigma_L / 255.0) * 0.2)

        # C3 — FSGD (zone 0–40 %)
        f2 = max(1, int(height * 0.40))
        facial = roi_gray[0:f2, :]
        if facial.size == 0:
            C3 = 0.0
        else:
            edges     = cv2.Canny(facial.astype(np.uint8), 50, 150)
            rho_edges = float(np.sum(edges > 0)) / edges.size
            R_d       = float(np.sum(facial < 100)) / facial.size
            C3 = min(1.0, rho_edges * 2.0 + R_d * 0.5)

        # C2 — FTDD (ROI entière)
        lap     = cv2.Laplacian(roi_gray.astype(np.float64), cv2.CV_64F)
        C2_flag = float(lap.var()) > 500.0

        return float(C1), float(C3), bool(C2_flag)

    except Exception as e:
        st.warning(f"[extraire_caracteristiques] {e}")
        return 0.0, 0.0, False


# ═══════════════════════════════════════════════════════════════
# ANNOTATION IMAGE
# ═══════════════════════════════════════════════════════════════
def dessiner_detection(img, det, classif_finale, zone):
    if det["DL_Classification"] == "Aucun Objet Détecté" or "bbox" not in det:
        return img
    x1, y1, x2, y2 = det["bbox"]
    h = y2 - y1; w = x2 - x1

    if zone == "Zone 2" and det["DL_Classification"] == "Gorille_Montagne":
        col = (255, 60, 60)
    elif classif_finale == "Gorille_Montagne":
        col = (50, 205, 50)
    else:
        col = (255, 165, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)

    c1c = (0, 220, 0) if det["C1"] >= ExpertKnowledge.SEUIL_C1 else (255, 80, 80)
    cv2.rectangle(img, (x1 + 15, y1 + int(h * 0.20)), (x2 - 15, y1 + int(h * 0.60)), c1c, 2)
    cv2.putText(img, f"C1:{det['C1']:.2f}", (x1 + 18, y1 + int(h * 0.20) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, c1c, 2)

    c3c = (80, 80, 255) if det["C3"] >= ExpertKnowledge.SEUIL_C3 else (255, 80, 80)
    cv2.rectangle(img, (x1 + int(w * 0.25), y1 + 8), (x2 - int(w * 0.25), y1 + int(h * 0.40)), c3c, 2)
    cv2.putText(img, f"C3:{det['C3']:.2f}", (x1 + int(w * 0.25) + 5, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, c3c, 2)

    hdr_y1 = max(0, y1 - 65)
    if hdr_y1 == 0:
        hdr_y1 = y2 + 4
    label_w = max(300, w)
    cv2.rectangle(img, (x1, hdr_y1), (x1 + label_w, hdr_y1 + 62), col, -1)
    cv2.putText(img, classif_finale, (x1 + 8, hdr_y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(img,
                f"{zone} | YOLO:{det['S0_general']:.0%} | S_E:{det['S_E']:.2f}",
                (x1 + 8, hdr_y1 + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return img


# ═══════════════════════════════════════════════════════════════
# PIPELINE COMPLET : YOLO → MVH → FUSION
# ═══════════════════════════════════════════════════════════════
def run_inference(image_pil, model):
    try:
        image_array = np.array(image_pil.convert("RGB"), dtype=np.uint8)
        h0, w0 = image_array.shape[:2]
        if max(h0, w0) > 1280:
            scale = 1280 / max(h0, w0)
            image_array = cv2.resize(
                image_array,
                (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_AREA,
            )

        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        results   = model(image_bgr, conf=0.25, device=DEVICE, verbose=False)

        if results and len(results[0].boxes) > 0:
            box       = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            classe_id = int(box.cls[0].cpu().numpy())
            confiance = float(box.conf[0].cpu().numpy())
            class_names = {0: "Autres_gorilles", 1: "Gorille_Montagne"}
            dl_classif  = class_names.get(classe_id, "Gorille_Montagne")

            C1, C3, C2_flag = extraire_caracteristiques(image_array, (x1, y1, x2, y2))
            S_E = ExpertKnowledge.calculer_SE(C1, C2_flag, C3)

            det = {
                "DL_Classification": dl_classif,
                "S0_general": confiance,
                "C1": C1, "C3": C3, "C2_flag": C2_flag,
                "S_E": S_E,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "classe_id": classe_id,
            }
        else:
            det = {
                "DL_Classification": "Aucun Objet Détecté",
                "S0_general": 0.0,
                "C1": 0.0, "C3": 0.0, "C2_flag": False,
                "S_E": 0.0,
            }

        classif_finale, _, _, zone, _, _ = ExpertKnowledge.valider_classification(
            det["DL_Classification"], det["C1"], det["C3"],
            det["C2_flag"], score_ia=det["S0_general"],
        )
        img_annotee = dessiner_detection(image_array.copy(), det, classif_finale, zone)
        return img_annotee, det

    except Exception as e:
        st.error(f"[run_inference] {e}\n{traceback.format_exc()}")
        return None, {"erreur": str(e)}


# ═══════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════
def main():
    st.title("🦍 Système Expert — Gorilles de Montagne")
    st.markdown(
        """
        ### YOLOv11n + Module de Validation Heuristique (MVH)
        > Résultats en **3 étapes** : 🤖 YOLO → 🧠 MVH → 🏆 Décision finale

        | Formule | Description |
        |---|---|
        | `C₁ = 0.4·(L̄/255) + 0.4·R_b + 0.2·(σ_L/255)` | DCID — Dos Argenté |
        | `Var(ΔI) > 500` | FTDD — Densité du Pelage |
        | `C₃ = min(1, 2·ρ_contours + 0.5·R_d)` | FSGD — Morphologie Faciale |
        | `S_E = 0.40·C₁ + 0.20·C₂ + 0.40·C₃` | Score Expert |
        | `Decision = 0.60·ScoreIA + 0.40·S_E ≥ Γ=0.50` | Fusion décisionnelle |
        """
    )
    st.divider()

    # ── Chargement du modèle ───────────────────────────────────
    model = load_model()
    if model is None:
        st.stop()

    st.success(f"✅ Modèle chargé sur **{DEVICE.upper()}**")

    # ── Upload image ───────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📷 Uploadez une image de gorille",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Veuillez uploader une image pour lancer l'analyse.")
        return

    image_pil = Image.open(uploaded_file).convert("RGB")

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("📷 Image d'entrée")
        st.image(image_pil, use_container_width=True)

    # ── Inférence ─────────────────────────────────────────────
    with st.spinner("🔍 Analyse en cours…"):
        image_annot, det = run_inference(image_pil, model)

    with col_out:
        st.subheader("🔍 Résultat annoté — décision FINALE · zones C₁(vert) C₃(bleu)")
        if image_annot is not None:
            st.image(image_annot, use_container_width=True)

    if "erreur" in det:
        st.error(f"❌ {det['erreur']}")
        return

    # ── Aucune détection ──────────────────────────────────────
    if det["DL_Classification"] == "Aucun Objet Détecté":
        st.warning("⚪ Aucun gorille détecté par YOLO.")
        return

    # ── Fusion MVH ────────────────────────────────────────────
    classif_finale, S_E, decision, zone, explication, preuves = \
        ExpertKnowledge.valider_classification(
            det["DL_Classification"],
            det["C1"], det["C3"], det["C2_flag"],
            score_ia=det["S0_general"],
        )

    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 1 — YOLO
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🤖 Étape 1 — Ce que YOLO a décidé *(avant MVH)*")

    yolo_badge = {
        "Gorille_Montagne": "🟢 Gorille_Montagne",
        "Autres_gorilles":  "🟡 Autres_gorilles",
    }.get(det["DL_Classification"], det["DL_Classification"])

    yolo_stat = (
        ("✅ CONFIANT" if det["S0_general"] >= ExpertKnowledge.THETA_CONF else "⚠️ INCERTAIN")
        + f" (θ_conf={ExpertKnowledge.THETA_CONF})"
    )

    c1e, c2e, c3e = st.columns(3)
    c1e.metric("Classe prédite par YOLO", yolo_badge)
    c2e.metric("Score de confiance (ScoreIA)", f"{det['S0_general']:.1%}")
    c3e.metric(f"Statut vis-à-vis θ_conf={ExpertKnowledge.THETA_CONF}", yolo_stat)

    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 2 — MVH
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧠 Étape 2 — Ce que le MVH a calculé")

    c1_str = f"{det['C1']:.3f}  {'✅' if det['C1'] >= ExpertKnowledge.SEUIL_C1 else '❌'}  (seuil ≥ {ExpertKnowledge.SEUIL_C1})"
    c2_str = f"{'Oui ✅' if det['C2_flag'] else 'Non ❌'}  (Var. Laplacien > 500)"
    c3_str = f"{det['C3']:.3f}  {'✅' if det['C3'] >= ExpertKnowledge.SEUIL_C3 else '❌'}  (seuil ≥ {ExpertKnowledge.SEUIL_C3})"
    se_str = f"{S_E:.3f}  →  {'✅ ≥ θ_veto' if S_E >= ExpertKnowledge.THETA_VETO else '❌ < θ_veto'}  ({ExpertKnowledge.THETA_VETO})"

    col_z, col_c1, col_c2 = st.columns(3)
    col_z.metric("🗺️ Zone décisionnelle", zone)
    col_c1.metric("C₁ — DCID (Dos Argenté)", c1_str)
    col_c2.metric("C₂ — FTDD (Texture Pelage)", c2_str)

    col_c3, col_se, _ = st.columns(3)
    col_c3.metric("C₃ — FSGD (Morphologie Faciale)", c3_str)
    col_se.metric("S_E = 0.40·C₁ + 0.20·C₂ + 0.40·C₃", se_str)

    # Tableau des preuves
    st.markdown("#### 🔬 Tableau des preuves MVH")
    tbl_data = []
    for p in preuves:
        s = f"{p['score']:.3f}" if isinstance(p["score"], float) else str(p["score"])
        tbl_data.append({
            "Descripteur":  p["nom"],
            "Zone ROI":     p["zone"],
            "Score":        s,
            "Seuil":        str(p["seuil"]),
            "Poids":        p["poids"],
            "Contribution": f"{p['contribution']:.3f}",
            "✓/✗":         "✅" if p["validee"] else "❌",
        })
    tbl_data.append({
        "Descripteur":  "**S_E total**",
        "Zone ROI":     "—",
        "Score":        f"**{S_E:.3f}**",
        "Seuil":        f"θ_veto={ExpertKnowledge.THETA_VETO}",
        "Poids":        "—",
        "Contribution": "—",
        "✓/✗":         "✅" if S_E >= ExpertKnowledge.THETA_VETO else "❌",
    })
    st.table(tbl_data)

    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 3 — Décision finale
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🏆 Étape 3 — Décision finale *(après fusion/véto)*")

    final_badge = {
        "Gorille_Montagne": "🟢 GORILLE DE MONTAGNE",
        "Autres_gorilles":  "🟡 AUTRE GORILLE",
    }.get(classif_finale, "⚪ AUCUNE DÉTECTION")

    if classif_finale == det["DL_Classification"]:
        delta = "✅ Confirmé par MVH"
    elif zone == "Zone 2":
        delta = "⚠️ VÉTO MVH — YOLO contredit"
    else:
        delta = "🔄 Corrigé par fusion (Zone 3)"

    if zone == "Zone 3":
        dec_str = (
            f"{ExpertKnowledge.ALPHA_IA}×{det['S0_general']:.3f} + "
            f"{ExpertKnowledge.ALPHA_MVH}×{S_E:.3f} = {decision:.4f}  "
            f"({'≥' if decision >= ExpertKnowledge.GAMMA else '<'} Γ={ExpertKnowledge.GAMMA})"
        )
    elif zone == "Zone 1":
        dec_str = f"YOLO direct — ScoreIA={det['S0_general']:.3f} (MVH confirme S_E={S_E:.3f})"
    else:
        dec_str = f"Véto MVH — S_E={S_E:.3f} < θ_veto={ExpertKnowledge.THETA_VETO}"

    cf1, cf2, cf3 = st.columns(3)
    cf1.metric("Classification finale", final_badge)
    cf2.metric("Changement vs YOLO", delta)
    cf3.metric("Calcul de décision", dec_str)

    st.info(f"💬 **Explication :** {explication}")

    # ── Export JSON ───────────────────────────────────────────
    with st.expander("💾 Export JSON"):
        rapport = json.dumps(
            {
                "date": datetime.now().isoformat(),
                "seed": 42,
                "etape1_yolo": {
                    "classe": det["DL_Classification"],
                    "confiance": det["S0_general"],
                    "statut": yolo_stat,
                },
                "etape2_mvh": {
                    "C1": det["C1"], "C2_flag": det["C2_flag"],
                    "C3": det["C3"], "S_E": S_E, "zone": zone,
                },
                "etape3_final": {
                    "classification": classif_finale,
                    "decision": decision,
                    "delta": delta,
                    "explication": explication,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        st.code(rapport, language="json")
        st.download_button(
            "⬇️ Télécharger le rapport JSON",
            data=rapport,
            file_name=f"rapport_gorille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    st.divider()
    st.caption("**Auteur :** KANDOLO Herman · IS–VNU · hermankandolo2022@gmail.com")


if __name__ == "__main__":
    main()
