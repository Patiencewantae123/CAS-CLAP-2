# =============================================================================
# CAS-V: Conflict-Aware Semantic Verification
#
# 1. CLIP zero-shot recognition: scene, indoor/outdoor, per-concept evidence.
# 2. Verification of every candidate sound concept:
#       score = w_visual * visual_evidence + w_scene * scene_support
#               - conflict_penalty * n_conflicts
#    -> ACCEPT / REPAIR / REJECT per concept.
# 3. Repair: if too few concepts are accepted, fill from the recognised
#    scene's canonical concepts that have no explicit conflict.
#
# The unverified baseline is the top-k concepts by raw CLIP similarity, which
# isolates the contribution of verification in the comparison.
# =============================================================================

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .concepts import CONCEPTS, SCENES, ENV_PROMPTS

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@dataclass
class VerifierConfig:
    w_visual: float = 0.45
    w_scene: float = 0.55
    conflict_penalty: float = 0.35
    accept_threshold: float = 0.55
    repair_threshold: float = 0.35
    env_conflict_confidence: float = 0.70
    uncertainty_threshold: float = 0.35
    max_concepts: int = 4
    min_concepts: int = 2


@dataclass
class ConceptVerdict:
    concept: str
    visual_evidence: float
    scene_support: float
    conflicts: list = field(default_factory=list)
    score: float = 0.0
    decision: str = "REJECT"


class CLIPRecognizer:
    def __init__(self, model_id=CLIP_MODEL_ID, device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self._text_cache = {}

    @torch.no_grad()
    def _text_embeddings(self, texts):
        key = tuple(texts)
        if key not in self._text_cache:
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
            emb = self.model.get_text_features(**inputs)
            if not torch.is_tensor(emb):
                emb = emb.pooler_output
            self._text_cache[key] = torch.nn.functional.normalize(emb, dim=-1)
        return self._text_cache[key]

    @torch.no_grad()
    def image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        if not torch.is_tensor(emb):
            emb = emb.pooler_output
        return torch.nn.functional.normalize(emb, dim=-1)[0]

    def similarities(self, image_emb, texts):
        return (self._text_embeddings(texts) @ image_emb).cpu().numpy()

    def probabilities(self, image_emb, texts):
        logits = self.model.logit_scale.exp().item() * self.similarities(image_emb, texts)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CASVerifier:
    def __init__(self, recognizer, config=None):
        self.recognizer = recognizer
        self.config = config or VerifierConfig()
        self.scene_names = list(SCENES)
        self.concept_names = list(CONCEPTS)

    def recognise(self, image):
        image_emb = self.recognizer.image_embedding(image)

        scene_probs = self.recognizer.probabilities(
            image_emb, [SCENES[name]["prompt"] for name in self.scene_names])
        env_probs = self.recognizer.probabilities(
            image_emb, [ENV_PROMPTS["indoor"], ENV_PROMPTS["outdoor"]])
        concept_sims = self.recognizer.similarities(
            image_emb, [f"a photo of {CONCEPTS[name]['cue']}" for name in self.concept_names])

        order = np.argsort(scene_probs)[::-1]
        return {
            "scene_probs": dict(zip(self.scene_names, scene_probs.tolist())),
            "top_scene": self.scene_names[order[0]],
            "scene_confidence": float(scene_probs[order[0]]),
            "scene_margin": float(scene_probs[order[0]] - scene_probs[order[1]]),
            "env_probs": {"indoor": float(env_probs[0]), "outdoor": float(env_probs[1])},
            "concept_sims": dict(zip(self.concept_names, concept_sims.tolist())),
        }

    def _verify_concept(self, name, recognition, visual_z):
        cfg = self.config
        scene_probs = recognition["scene_probs"]
        env_probs = recognition["env_probs"]
        top_scene = recognition["top_scene"]

        scene_support = sum(p for s, p in scene_probs.items() if name in SCENES[s]["concepts"])

        conflicts = []
        concept_env = CONCEPTS[name]["env"]
        if concept_env == "indoor" and env_probs["outdoor"] >= cfg.env_conflict_confidence:
            conflicts.append(f"'{name}' is an indoor sound but the image is outdoors "
                             f"(p={env_probs['outdoor']:.2f}).")
        if concept_env == "outdoor" and env_probs["indoor"] >= cfg.env_conflict_confidence:
            conflicts.append(f"'{name}' is an outdoor sound but the image is indoors "
                             f"(p={env_probs['indoor']:.2f}).")
        scene_env = SCENES[top_scene]["env"]
        if (recognition["scene_confidence"] >= 0.5 and scene_env != "any"
                and concept_env not in ("any", scene_env)):
            conflicts.append(f"'{name}' ({concept_env}) contradicts the recognised "
                             f"{scene_env} scene '{top_scene}'.")

        visual = float(_sigmoid(visual_z[name]))
        score = cfg.w_visual * visual + cfg.w_scene * scene_support - cfg.conflict_penalty * len(conflicts)
        score = float(np.clip(score, 0.0, 1.0))

        if conflicts:
            decision = "REJECT"
        elif score >= cfg.accept_threshold:
            decision = "ACCEPT"
        elif score >= cfg.repair_threshold:
            decision = "REPAIR"
        else:
            decision = "REJECT"

        return ConceptVerdict(name, visual, float(scene_support), conflicts, score, decision)

    def verify(self, image):
        cfg = self.config
        recognition = self.recognise(image)

        sims = np.array([recognition["concept_sims"][n] for n in self.concept_names])
        z = (sims - sims.mean()) / (sims.std() + 1e-8)
        visual_z = dict(zip(self.concept_names, z))

        verdicts = sorted(
            (self._verify_concept(n, recognition, visual_z) for n in self.concept_names),
            key=lambda v: v.score, reverse=True)

        verified = [v.concept for v in verdicts if v.decision == "ACCEPT"][:cfg.max_concepts]
        repaired = []
        if len(verified) < cfg.min_concepts:
            by_name = {v.concept: v for v in verdicts}
            for name in SCENES[recognition["top_scene"]]["concepts"]:
                if len(verified) >= cfg.max_concepts:
                    break
                if name not in verified and not by_name[name].conflicts:
                    verified.append(name)
                    repaired.append(name)

        if recognition["scene_confidence"] < cfg.uncertainty_threshold:
            status = "UNCERTAIN"
        elif repaired:
            status = "REPAIR"
        elif verified:
            status = "ACCEPT"
        else:
            status = "REJECT"

        baseline = sorted(self.concept_names, key=lambda n: recognition["concept_sims"][n],
                          reverse=True)[:cfg.max_concepts]

        return {
            "recognition": recognition,
            "verdicts": [v.__dict__ for v in verdicts],
            "verified_concepts": verified,
            "repaired_concepts": repaired,
            "status": status,
            "caption": SCENES[recognition["top_scene"]]["caption"],
            "baseline_concepts": baseline,
        }


def load_image(path):
    return Image.open(path).convert("RGB")
