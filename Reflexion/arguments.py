from argparse import ArgumentParser

class GroupParams:
    '''参数组容器'''
    pass

class ParamGroup:
    '''参数组基类'''
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

class ModelParams(ParamGroup): 
    '''高斯模型参数组'''
    def __init__(self, parser):
        self.scene_path = "..\\..\\data\\example\\train"    #场景数据路径
        self.model_path = "..\\..\\output\\example\\train"  #模型保存路径

        self.pretrained_model_path="..\\..\\data\\example\\train.ply"       #预训练模型路径

        self.sh_degree = 3      #最大SH阶数
        self.resolution = -1    #分辨率控制
        self.white_background = False      #是否使用白色背景代替默认的黑色

        self.data_device = "cuda"   #数据设备   
        super().__init__(parser, "Model Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.scene_path = os.path.abspath(g.scene_path)
        return g