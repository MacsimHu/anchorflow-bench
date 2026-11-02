"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.


class Registry: # 定义Registry类，用于存储所有注册的类
    r"""Registry is a class that stores class types across the entire dataset.
    It makes coding very convenient and flexible. Should not be manually
    instantiated.
    """

    mapping = { # 定义Registry类的属性mapping，存储所有注册的类
        "dataset_name_mapping": {}, # 存储所有注册的数据集类
        "model_name_mapping": {},
        "trainer_name_mapping": {},
        "loss_name_mapping": {},
    }
    loaded = False # 定义Registry类的属性loaded，用于判断是否加载了所有注册的类

    @classmethod # 定义Registry类的方法register_dataset，用于注册数据集类
    def register_dataset(cls, name): # name是数据集类的名称，用于在注册表中唯一标识数据集类，"pfdeltadata"
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_dataset("opfdata")
            class OPFData:
                ...
        """

        def wrap(func): #func是PFDeltaDataset类
            cls.mapping["dataset_name_mapping"][name] = func # 注册数据集类
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_model("powerflownet")
            class PowerFlowNet:
                ...
        """

        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod # 标记这是一个类方法，调用时不需要实例化 Registry 类
    def register_trainer(cls, name): # register动词注册。接收一个参数 name ，作为训练器在注册表中的唯一标识
        # cls是Registry类的引用，用于访问类属性和方法，这里用于访问类属性 mapping， 
        # 用于存储所有注册的训练器类， 键为 name，值为训练器类，
        # 下面用于存储所有注册的损失函数类， 键为 name，值为损失函数类，
        r"""Register a trainer to registry with key 'name' # r是为了在文档字符串中显示原始字符串, 避免转义字符的问题

        Args:
            name: Key with which the trainer will be registered.

        Usage: 

            from core.utils.registry import registry

            @registry.register_trainer("base_trainer")
            class BaseTrainer:
                ...
        """
        # func是被装饰的 训练器类（GNN_Trainer），wrap是包装函数，用于将训练器类存储到注册表中
        def wrap(func): # 这是一个闭包函数，接收被装饰的 训练器类 func 。闭包 ： wrap 函数可以访问外部函数的 name 参数和 cls 参数
            cls.mapping["trainer_name_mapping"][name] = func #将训练器类 存储到注册表的映射中。cls.mapping 字典存储注册的类，实现全局访问
            return func #返回原始的训练器类，保持装饰器的透明性

        return wrap # 返回包装函数，这是装饰器的核心机制
        #注册表中现在包含了 "base_trainer"字符串 到 BaseTrainer 类的映射

        # 注册后，可以通过以下方式获取训练器类：
        #trainer_class = registry.get_trainer_class("base_trainer")

        # 创建训练器实例
        #trainer = trainer_class(config)

        # 使用训练器
        #trainer.train()

        # 在base_trainer.py中定义了BaseTrainer类：
        # @registry.register_trainer("base_trainer") 这里直接调用此文件的register_trainer函数
        # class BaseTrainer:    
            # def __init__(self, config):
                #self.config = config
                #print("GNNTrainer initialized")
    
            #def train(self):
                #print("Training with GNN...")
        
        #使用注册的训练器
        # 1. 从注册表获取训练器类
        #trainer_class = registry.get_trainer_class("base_trainer")

        # 2. 创建训练器实例
        #config = {"lr": 0.001, "batch_size": 32}
        #trainer = trainer_class(config)
        # 输出: GNNTrainer initialized

        # 3. 使用训练器
        #trainer.train()
        # 输出: Training with GNN...

# 这行代码执行时：
# 1. 先执行 registry.register_trainer("gnn_trainer")
# 2. 返回 wrap 函数
# 3. 然后执行 wrap(GNNTrainer)

# wrap(GNNTrainer) 执行时：
# 1. 获取 cls = Registry 类
# 2. 获取 name = "gnn_trainer"
# 3. 获取 func = GNNTrainer 类
# 4. 执行：cls.mapping["trainer_name_mapping"]["gnn_trainer"] = GNNTrainer，因为一个registery里面可以注册多种类型的类，单就训练器类而言也有多个训练器类
# 5. 返回 GNNTrainer 类本身

    @classmethod # 标记这是一个类方法，调用时不需要实例化 Registry 类
    def register_loss(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the loss will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_loss("obj_n_penalty")
            class Objective_n_Penalty:
                ...
        """

        def wrap(func): # 包装函数，用于将损失函数类存储到注册表中
            cls.mapping["loss_name_mapping"][name] = func
            return func # 返回原始的损失函数类，保持装饰器的透明性

        return wrap # 返回包装函数，这是装饰器的核心机制

    @classmethod
    def get_dataset_class(cls, name): # 获取注册的数据集类
        r"""Obtain registry dataset class with key 'name'

        Args:
            name: Key with which the dataset is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                OPFData = registry.get_dataset_class("opfdata")
                ...
        """

        return cls.mapping["dataset_name_mapping"].get(name, None) # 获取注册的数据集类，若不存在则返回None
        # cls.mapping["dataset_name_mapping"] 是一个字典，存储了所有注册的数据集类
        # cls是Registry类的实例，mapping是Registry类的属性，
        # mapping["dataset_name_mapping"]是一个字典，存储了所有注册的数据集类

    @classmethod
    def get_model_class(cls, name):
        r"""Obtain registry model class with key 'name'

        Args:
            name: Key with which the model is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                PowerFlowNet = registry.get_model_class("powerflownet")
                ...
        """

        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name):
        r"""Obtain registry trainer class with key 'name'

        Args:
            name: Key with which the model is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                BaseTrainer = registry.get_trainer_class("base_trainer")
                ...
        """
        # cls.mapping["trainer_name_mapping"] 是一个字典，存储了所有注册的训练器类
        # cls是Registry类的实例，mapping是Registry类的属性，
        # mapping["trainer_name_mapping"]是一个字典，存储了所有注册的训练器类
        return cls.mapping["trainer_name_mapping"].get(name, None) # 获取注册的训练器类，若不存在则返回None

    @classmethod
    def get_loss_class(cls, name): 
        # 获取注册的损失函数类
        # cls.mapping["loss_name_mapping"] 是一个字典，存储了所有注册的损失函数类
        # cls是Registry类的实例，mapping是Registry类的属性，
        # mapping["loss_name_mapping"]是一个字典，存储了所有注册的损失函数类
        r"""Obtain registry loss class with key 'name'

        Args:
            name: Key with which the loss is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                BaseTrainer = registry.get_loss_class("obj_n_penalty")
                ...
        """

        return cls.mapping["loss_name_mapping"].get(name, None)


registry = Registry()
