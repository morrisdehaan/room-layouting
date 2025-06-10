class_list = [
    "a wall",
    "a door",
    "a bed",
    "a chair",
    "a stool",
    "a nightstand",
    "a toilet",
    "a dressing table",
    "a wardrobe",
    "a tv cabinet",
    "a cupboard",
    "a chandelier",
    "a shower room",
    "curtains",
    "a carpet",
    "plants",
    "a wall decoration",
    # canonical classes
    "stuff",
    "things",
    "an object",
    "a texture"
]

def to_prompt(string):
    return [
        f"an image of {string}",
        f"a photo of {string} in a bedroom",
        f"a closeup of {string} in an indoor scene"
    ]

idx2prompts = {i: to_prompt(class_list[i]) for i in range(len(class_list))}

idx2class = {i: class_list[i] for i in range(len(class_list))}
