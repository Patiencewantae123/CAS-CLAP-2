# =============================================================================
# CAS-V: Conflict-Aware Semantic Verification
#
# 1. Recognition (CLIP zero-shot)
#      scene  : Places365 (365 classes); P(indoor) from the official IO labels
#      object : ImageNet (1000 classes)
#      source : visibility of each sound source in vocab.SOUND_SOURCES
# 2. Candidate retrieval: top-k AudioCaps train captions by image-text
#    similarity. The top-1 caption, unverified, is the baseline prompt.
# 3. Verification of every candidate caption
#      score = w_visual * visual + w_scene * scene_support
#              + w_object * object_support
#    with explicit conflicts (REJECT):
#      - environment: caption is indoor/outdoor-specific, image is the other
#      - hallucinated source: caption names a sound source the image lacks
# 4. Selection: best ACCEPT caption. Otherwise REPAIR with the best
#    conflict-free caption, or, if every candidate conflicts, a prompt built
#    from the recognised scene and object.
# =============================================================================

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import vocab

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CACHE_DIR = vocab.DATA_DIR / "cache"


@dataclass
class VerifierConfig:
    retrieve_k: int = 20
    support_top_n: int = 5
    w_visual: float = 0.40
    w_scene: float = 0.30
    w_object: float = 0.30
    accept_threshold: float = 0.60
    repair_threshold: float = 0.40
    env_conflict_confidence: float = 0.70
    source_absent_z: float = 0.0
    uncertainty_threshold: float = 0.10


class CLIPRecognizer:
    def __init__(self, model_id=CLIP_MODEL_ID, device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.logit_scale = self.model.logit_scale.exp().item()

    @staticmethod
    def _tensor(output):
        return output if torch.is_tensor(output) else output.pooler_output

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=256):
        key = hashlib.sha1(("\n".join(texts) + self.model_id).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"clip_text_{key}.npy"
        if cache.exists():
            return np.load(cache)
        chunks = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[start:start + batch_size], return_tensors="pt",
                                    padding=True, truncation=True, max_length=77).to(self.device)
            emb = self._tensor(self.model.get_text_features(**inputs))
            chunks.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        embeddings = np.concatenate(chunks).astype(np.float32)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, embeddings)
        return embeddings

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self._tensor(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1)[0].cpu().numpy()

    def softmax(self, sims):
        logits = self.logit_scale * sims
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()


def _minmax(values):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-8 else np.full_like(values, 0.5)


class CASVerifier:
    def __init__(self, recognizer, config=None):
        self.recognizer = recognizer
        self.config = config or VerifierConfig()

        self.scene_names, self.scene_envs = vocab.load_places365()
        self.object_names = vocab.load_imagenet()
        self.captions = vocab.load_audiocaps_captions("train")
        self.source_names = list(dict.fromkeys(vocab.SOUND_SOURCES.values()))

        print(f"[*] Vocabulary: {len(self.scene_names)} Places365 scenes, {len(self.object_names)} "
              f"ImageNet objects, {len(self.captions)} AudioCaps train captions")
        encode = recognizer.encode_texts
        self.scene_emb = encode([f"a photo of a {n}" for n in self.scene_names])
        self.object_emb = encode([f"a photo of a {n}" for n in self.object_names])
        self.source_emb = encode([f"a photo of {s}" for s in self.source_names])
        self.caption_emb = encode(self.captions)
        self.indoor_mask = np.array([env == "indoor" for env in self.scene_envs])

    def recognise(self, image):
        cfg = self.config
        image_emb = self.recognizer.encode_image(image)

        scene_probs = self.recognizer.softmax(self.scene_emb @ image_emb)
        object_probs = self.recognizer.softmax(self.object_emb @ image_emb)
        source_sims = self.source_emb @ image_emb
        source_z = (source_sims - source_sims.mean()) / (source_sims.std() + 1e-8)

        top_scenes = np.argsort(scene_probs)[::-1][:cfg.support_top_n]
        top_objects = np.argsort(object_probs)[::-1][:cfg.support_top_n]
        p_indoor = float(scene_probs[self.indoor_mask].sum())

        return {
            "image_emb": image_emb,
            "top_scenes": [(self.scene_names[i], float(scene_probs[i])) for i in top_scenes],
            "top_objects": [(self.object_names[i], float(object_probs[i])) for i in top_objects],
            "scene_idx": top_scenes,
            "scene_w": scene_probs[top_scenes],
            "object_idx": top_objects,
            "object_w": object_probs[top_objects],
            "env_probs": {"indoor": p_indoor, "outdoor": 1.0 - p_indoor},
            "source_z": dict(zip(self.source_names, source_z.tolist())),
        }

    def _conflicts(self, caption, recognition):
        cfg = self.config
        conflicts = []
        env = vocab.caption_environment(caption)
        env_probs = recognition["env_probs"]
        if env != "any":
            other = "outdoor" if env == "indoor" else "indoor"
            if env_probs[other] >= cfg.env_conflict_confidence:
                conflicts.append(f"{env} sound, but image is {other} (p={env_probs[other]:.2f})")
        for source in vocab.mentioned_sources(caption):
            z = recognition["source_z"][source]
            if z < cfg.source_absent_z:
                conflicts.append(f"names '{source}', not visible in image (z={z:+.2f})")
        return conflicts

    def verify(self, image):
        cfg = self.config
        rec = self.recognise(image)

        sims = self.caption_emb @ rec["image_emb"]
        candidates = np.argsort(sims)[::-1][:cfg.retrieve_k]
        cand_emb = self.caption_emb[candidates]

        visual = _minmax(sims[candidates])
        scene_support = _minmax(cand_emb @ self.scene_emb[rec["scene_idx"]].T @ rec["scene_w"]
                                / rec["scene_w"].sum())
        object_support = _minmax(cand_emb @ self.object_emb[rec["object_idx"]].T @ rec["object_w"]
                                 / rec["object_w"].sum())

        verdicts = []
        for j, idx in enumerate(candidates):
            caption = self.captions[idx]
            conflicts = self._conflicts(caption, rec)
            score = float(cfg.w_visual * visual[j] + cfg.w_scene * scene_support[j]
                          + cfg.w_object * object_support[j])
            if conflicts:
                decision = "REJECT"
            elif score >= cfg.accept_threshold:
                decision = "ACCEPT"
            elif score >= cfg.repair_threshold:
                decision = "REPAIR"
            else:
                decision = "REJECT"
            verdicts.append({
                "caption": caption,
                "clip_similarity": float(sims[idx]),
                "visual": float(visual[j]),
                "scene_support": float(scene_support[j]),
                "object_support": float(object_support[j]),
                "score": score,
                "conflicts": conflicts,
                "decision": decision,
            })

        baseline = verdicts[0]["caption"]
        ranked = sorted(verdicts, key=lambda v: v["score"], reverse=True)
        accepted = [v for v in ranked if v["decision"] == "ACCEPT"]
        clean = [v for v in ranked if not v["conflicts"]]

        top_scene, top_scene_p = rec["top_scenes"][0]
        top_object = rec["top_objects"][0][0]
        if accepted:
            prompt, status = accepted[0]["caption"], "ACCEPT"
        elif clean:
            prompt, status = clean[0]["caption"], "REPAIR"
        else:
            prompt, status = f"ambient sounds of a {top_scene} with a {top_object}", "REPAIR"
        if max(top_scene_p, rec["top_objects"][0][1]) < cfg.uncertainty_threshold:
            status = "UNCERTAIN"

        visible_sources = [s for s, z in sorted(rec["source_z"].items(), key=lambda kv: -kv[1])[:3]]
        return {
            "top_scenes": rec["top_scenes"],
            "top_objects": rec["top_objects"],
            "env_probs": rec["env_probs"],
            "visible_sources": visible_sources,
            "verdicts": verdicts,
            "baseline_prompt": baseline,
            "casv_prompt": prompt,
            "status": status,
            "n_rejected": sum(v["decision"] == "REJECT" for v in verdicts),
        }


def load_image(path):
    return Image.open(path).convert("RGB")
