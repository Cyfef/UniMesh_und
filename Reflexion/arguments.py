from argparse import ArgumentParser
from agents import ReflexionStrategy,EvaluatorModel

class GroupParams:
    '''Parameter group container'''
    pass

class ParamGroup:
    '''Parameter Group Base Class'''
    def __init__(self, parser: ArgumentParser, name : str):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            t = type(value)
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) :
                setattr(group, arg[0], arg[1])
        return group

CAPTION_PROMPT="""
You are given 6 renderings of a 3D object, please generate a concise caption that describes it. Captions should typically begin with an article ("a" or "an"), followed by color(s), shape, and the object type.Include distinctive features introduced by "with" when relevant (e.g., parts, textures, accessories). Use simple, everyday vocabulary and mention colors, materials (wooden, metal, plastic, etc.), and any notable details like wheels, windows, eyes, or decorations. Avoid long or complex sentences. The caption should be a short phrase or a simple sentence that captures the essential visual attributes.
"""

class ReflexionParams(ParamGroup): 
    '''Reflexion Parameters Group'''
    def __init__(self, parser):
        self.n:int=3
        self.prompt:str=CAPTION_PROMPT
        self.strategy:ReflexionStrategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
        self.Evaluator:EvaluatorModel=EvaluatorModel.BAGEL
        
        self.objs_dir:str="../data/Cap3D_imgs"
        self.save_dir:str="../Captions"
          
        super().__init__(parser, "Reflexion Parameters")

    def extract(self, args):
        g = super().extract(args)
        return g