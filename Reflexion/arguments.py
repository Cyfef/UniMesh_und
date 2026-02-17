from argparse import ArgumentParser
from agents import CoTAgent, ReflexionStrategy

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


CAPTION_PROMPT=""


class ReflexionParams(ParamGroup): 
    '''Reflexion Parameters Group'''
    def __init__(self, parser):
        self.prompt:str=CAPTION_PROMPT,  
        self.n:int=3,
        self.strategy:ReflexionStrategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        self.objs_dir:str=""
          
        super().__init__(parser, "Reflexion Parameters")

    def extract(self, args):
        g = super().extract(args)
        return g