CAPTION_PROMPT="""
You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
"""

COT=f"""
Images:<Image1><Image2><Image3><Image4><Image5><Image6>(In fact, these six images should be given to you, but here are omited.)
Prompt:{CAPTION_PROMPT}
Thought:The object is consistently shown with four slender, metallic legs across all views, and its overall form—featuring a backrest, armrests, and a seat cushion—matches the functional and structural definition of a sofa (or armchair-style sofa), distinguishing it from stools or ottomans which lack back/arm support.
Action: Finish[a sofa with four legs]
"""


COT_REFLECT=f"""
Images:<Image1><Image2><Image3><Image4><Image5><Image6>(In fact, these six images should be given to you, but here are omited.)
Prompt:{CAPTION_PROMPT}
Thought:its curved, symmetrical metal band resembles a classic double-pronged hair clip, and the central “panel” looked like a decorative embellishment.
Action: Finish["a hairpin"]

Reflection:My misidentification likely stemmed from overemphasizing the object’s curved, symmetrical form while underweighting functional cues (e.g., digital displays, buttons, wrist-scale context); to prevent recurrence, I will prioritize *functional affordances* and *scale-relative features* over superficial shape analogies when interpreting 3D renders.

"""