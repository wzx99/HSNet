import importlib
from collections import OrderedDict

import anyconfig
import munch

# yaml文件：
#   import： 使用的base yaml文件
#   package： 需要使用的包
#   define：list,其中每个元素为dict，
#        其中，name仅表示其在config加载过程中的名称用于给子yaml文件调用
#            class表示代码中对应class的名称
#            其余项表示初始化该class的参数，可继续嵌套base yaml的name或任意class
class Config(object):
    def __init__(self):
        pass

    def load(self, conf):
        conf = anyconfig.load(conf)
        return munch.munchify(conf)

    # 返回dict用于Configurable加载并初始化所需要的class
    # 将yaml读取为dict，其中class增加其引用路径，与base yaml文件中name相同的value替换为对应包含详细内容的dict
    def compile(self, conf, return_packages=False):
        packages = conf.get('package', [])
        defines = {}

        for path in conf.get('import', []):  # 加载base yaml文件
            parent_conf = self.load(path)
            parent_packages, parent_defines = self.compile(
                parent_conf, return_packages=True)
            packages.extend(parent_packages)
            defines.update(parent_defines)

        modules = []
        for package in packages:
            module = importlib.import_module(package)  #import 相关包
            modules.append(module)

        if isinstance(conf['define'], dict):
            conf['define'] = [conf['define']]

        for define in conf['define']:
            name = define.copy().pop('name')

            if not isinstance(name, str):
                raise RuntimeError('name must be str')
            
            # 1.将class添加对应包的路径 2.根据name将base yaml文件中加载的class dict替换到对应的位置
            defines[name] = self.compile_conf(define, defines, modules)  
            
        if return_packages:
            return packages, defines
        else:
            return defines

    def compile_conf(self, conf, defines, modules):
    # 1. 将class加入对应的module路径
    # 2. 以^开头的从已加载的define中寻找对应的name覆盖
    # 3. 没有class则不变
        if isinstance(conf, (int, float)):
            return conf
        elif isinstance(conf, str):
            if conf.startswith('^'):
                return defines[conf[1:]]
            if conf.startswith('$'):
                return {'class': self.find_class_in_modules(conf[1:], modules)}
            return conf
        elif isinstance(conf, dict):
            if 'class' in conf:
                conf['class'] = self.find_class_in_modules(
                    conf['class'], modules)
            if 'base' in conf:
                base = conf.copy().pop('base')

                if not isinstance(base, str):
                    raise RuntimeError('base must be str')

                conf = {
                    **defines[base],
                    **conf,
                }
            return {key: self.compile_conf(value, defines, modules) for key, value in conf.items()}
        elif isinstance(conf, (list, tuple)):
            return [self.compile_conf(value, defines, modules) for value in conf]
        else:
            return conf

    def find_class_in_modules(self, cls, modules):
        if not isinstance(cls, str):
            raise RuntimeError('class name must be str')

        if cls.find('.') != -1:
            package, cls = cls.rsplit('.', 1)
            module = importlib.import_module(package)
            if hasattr(module, cls):
                return module.__name__ + '.' + cls

        for module in modules:
            if hasattr(module, cls):
                return module.__name__ + '.' + cls
        raise RuntimeError('class not found ' + cls)


class State:
    def __init__(self, autoload=True, default=None):
        self.autoload = autoload
        self.default = default


class StateMeta(type):
    def __new__(mcs, name, bases, attrs):
        current_states = []
        for key, value in attrs.items():
            if isinstance(value, State):
                current_states.append((key, value))

        current_states.sort(key=lambda x: x[0])
        attrs['states'] = OrderedDict(current_states)
        new_class = super(StateMeta, mcs).__new__(mcs, name, bases, attrs)

        # Walk through the MRO
        states = OrderedDict()
        for base in reversed(new_class.__mro__):
            if hasattr(base, 'states'):
                states.update(base.states)
        new_class.states = states

        for key, value in states.items():
            setattr(new_class, key, value.default)

        return new_class


# 根据config得到的dict加载并初始化对应目标class
#  cls(**args)可直接根据args字典中的key传入参数
# 或目标class继承了Configurable，预先定义为State()的变量可根据init函数中的 self.load_all(**kwargs) 下根据dict进行初始化
# 或直接使用get函数从dict中获取并赋值进行初始化
class Configurable(metaclass=StateMeta):
    def __init__(self, *args, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

    @staticmethod
    def construct_class_from_config(args):
        cls = Configurable.extract_class_from_args(args)  #import并获取对应的class
        return cls(**args)  #根据config(dict)初始化class

    @staticmethod
    def extract_class_from_args(args):
        cls = args.copy().pop('class')
        package, cls = cls.rsplit('.', 1)
        module = importlib.import_module(package)
        cls = getattr(module, cls)
        return cls

    def load_all(self, **kwargs):  #将定义为state的变量进行初始化
        for name, state in self.states.items(): 
            if state.autoload:
                self.load(name, **kwargs)

    def load(self, state_name, **kwargs):
        # FIXME: kwargs should be filtered
        # Args passed from command line
        cmd = kwargs.pop('cmd', dict())
        if state_name in kwargs:
            setattr(self, state_name, self.create_member_from_config(
                (kwargs[state_name], cmd)))  #根据dict中的定义初始化变量
        else:
            setattr(self, state_name, self.states[state_name].default)  #dict中没有的使用定义时State(default=XXX)中的default初始化

    def create_member_from_config(self, conf):
        args, cmd = conf
        if args is None or isinstance(args, (int, float, str)):  #传入数字或字符串
            return args
        elif isinstance(args, (list, tuple)):
            return [self.create_member_from_config((subargs, cmd)) for subargs in args]  #传入列表
        elif isinstance(args, dict):
            if 'class' in args:  #若为class，则先初始化class再将其传入
                cls = self.extract_class_from_args(args)
                return cls(**args, cmd=cmd)
            return {key: self.create_member_from_config((subargs, cmd)) for key, subargs in args.items()} #传入dict
        else:
            return args

    def dump(self):
        state = {}
        state['class'] = self.__class__.__module__ + \
            '.' + self.__class__.__name__
        for name, value in self.states.items():
            obj = getattr(self, name)
            state[name] = self.dump_obj(obj)
        return state

    def dump_obj(self, obj):
        if obj is None:
            return None
        elif hasattr(obj, 'dump'):
            return obj.dump()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.dump_obj(value) for value in obj]
        elif isinstance(obj, dict):
            return {key: self.dump_obj(value) for key, value in obj.items()}
        else:
            return str(obj)

