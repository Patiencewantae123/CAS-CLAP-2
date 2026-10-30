# =============================================================================
# CAS-V concept vocabulary
#
# A scene is what CLIP recognises in the image. A sound concept is what we ask
# the audio generator to produce. Each concept carries the visual evidence that
# must be present in the image for it to be plausible ("cue") and the
# acoustic environment it belongs to ("env"), which the verifier uses to
# detect explicit conflicts.
# =============================================================================

# env: "indoor", "outdoor" or "any"
CONCEPTS = {
    # Busy road
    "traffic": {"cue": "a road full of cars in traffic", "env": "outdoor"},
    "engine sounds": {"cue": "cars and vehicles with engines", "env": "any"},
    "car movement": {"cue": "cars driving on a road", "env": "outdoor"},
    "urban ambience": {"cue": "a city street", "env": "outdoor"},
    # Office
    "keyboard typing": {"cue": "hands typing on a computer keyboard", "env": "indoor"},
    "office ambience": {"cue": "an office desk with a computer", "env": "indoor"},
    "indoor environment": {"cue": "the inside of a room", "env": "indoor"},
    "mouse clicks": {"cue": "a hand on a computer mouse", "env": "indoor"},
    # Beach
    "ocean waves": {"cue": "ocean waves breaking on a beach", "env": "outdoor"},
    "seagulls": {"cue": "a seaside beach with sky", "env": "outdoor"},
    "shoreline": {"cue": "a sandy shoreline", "env": "outdoor"},
    "water ambience": {"cue": "open water", "env": "outdoor"},
    # Forest
    "wind": {"cue": "trees in an open landscape", "env": "outdoor"},
    "birds chirping": {"cue": "a green forest with trees where birds live", "env": "outdoor"},
    "nature ambience": {"cue": "a natural landscape", "env": "outdoor"},
    "forest environment": {"cue": "a forest", "env": "outdoor"},
    # Cloth scratch
    "cloth rubbing": {"cue": "hands rubbing a piece of cloth", "env": "any"},
    "fabric movement": {"cue": "fabric being moved", "env": "any"},
    "soft friction": {"cue": "a soft textile surface being scratched", "env": "any"},
    # Leaves
    "leaves crunching": {"cue": "dry autumn leaves on the ground being stepped on", "env": "outdoor"},
    "dry foliage": {"cue": "a pile of dry brown leaves", "env": "outdoor"},
    "movement on leaves": {"cue": "someone moving through fallen leaves", "env": "outdoor"},
    # Metal hit
    "metal impact": {"cue": "a hammer striking metal", "env": "any"},
    "metallic resonance": {"cue": "a metal object", "env": "any"},
    "sharp strike": {"cue": "a tool hitting a hard surface", "env": "any"},
    # Distractor scenes (give the verifier real alternatives to reject)
    "rain falling": {"cue": "rain falling", "env": "outdoor"},
    "thunder": {"cue": "a stormy sky with lightning", "env": "outdoor"},
    "crowd cheering": {"cue": "a large crowd of people at an event", "env": "any"},
    "applause": {"cue": "an audience clapping", "env": "indoor"},
    "dishes clattering": {"cue": "dishes and plates in a kitchen", "env": "indoor"},
    "sizzling pan": {"cue": "food cooking in a frying pan", "env": "indoor"},
    "cat meowing": {"cue": "a cat", "env": "any"},
    "cat purring": {"cue": "a cat resting", "env": "any"},
    "dog barking": {"cue": "a dog", "env": "any"},
    "bicycle bell": {"cue": "a bicycle", "env": "outdoor"},
    "bicycle chain clicking": {"cue": "a bicycle wheel and chain", "env": "any"},
}

SCENES = {
    "busy road": {
        "prompt": "a photo of a busy road with heavy traffic",
        "env": "outdoor",
        "concepts": ["traffic", "engine sounds", "car movement", "urban ambience"],
        "caption": "Traffic / engine sounds",
    },
    "office": {
        "prompt": "a photo of a person working at an office computer",
        "env": "indoor",
        "concepts": ["keyboard typing", "office ambience", "indoor environment", "mouse clicks"],
        "caption": "Keyboard typing / office ambience",
    },
    "beach": {
        "prompt": "a photo of a beach with ocean waves",
        "env": "outdoor",
        "concepts": ["ocean waves", "seagulls", "shoreline", "water ambience"],
        "caption": "Waves / seagulls",
    },
    "forest": {
        "prompt": "a photo of a forest with trees",
        "env": "outdoor",
        "concepts": ["wind", "birds chirping", "nature ambience", "forest environment"],
        "caption": "Wind / birds / nature ambience",
    },
    "cloth scratch": {
        "prompt": "a close-up photo of hands scratching or rubbing cloth",
        "env": "any",
        "concepts": ["cloth rubbing", "fabric movement", "soft friction"],
        "caption": "Cloth scratching",
    },
    "leaves": {
        "prompt": "a photo of dry autumn leaves being kicked or stepped on",
        "env": "outdoor",
        "concepts": ["leaves crunching", "dry foliage", "movement on leaves"],
        "caption": "Leaves crunching",
    },
    "metal hit": {
        "prompt": "a photo of a hammer hitting metal",
        "env": "any",
        "concepts": ["metal impact", "metallic resonance", "sharp strike"],
        "caption": "Metal hit sound",
    },
    "rainstorm": {
        "prompt": "a photo of a rainstorm",
        "env": "outdoor",
        "concepts": ["rain falling", "thunder", "wind"],
        "caption": "Rain / thunder",
    },
    "concert": {
        "prompt": "a photo of a concert audience",
        "env": "any",
        "concepts": ["crowd cheering", "applause"],
        "caption": "Crowd / applause",
    },
    "kitchen": {
        "prompt": "a photo of someone cooking in a kitchen",
        "env": "indoor",
        "concepts": ["sizzling pan", "dishes clattering", "indoor environment"],
        "caption": "Cooking / kitchen",
    },
    "cat": {
        "prompt": "a photo of a cat",
        "env": "any",
        "concepts": ["cat meowing", "cat purring"],
        "caption": "Cat meowing / purring",
    },
    "dog": {
        "prompt": "a photo of a dog",
        "env": "any",
        "concepts": ["dog barking"],
        "caption": "Dog barking",
    },
    "bicycle": {
        "prompt": "a photo of a bicycle",
        "env": "outdoor",
        "concepts": ["bicycle bell", "bicycle chain clicking", "urban ambience"],
        "caption": "Bicycle",
    },
}

ENV_PROMPTS = {
    "indoor": "a photo taken indoors",
    "outdoor": "a photo taken outdoors",
}
