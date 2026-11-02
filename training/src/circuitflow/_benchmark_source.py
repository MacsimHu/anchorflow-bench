# SPDX-License-Identifier: MIT
# Adapted mechanically from the public PFDelta parser; see licenses/benchmark-MIT.txt.
# Registry coupling was removed; parsing and split-selection expressions are unchanged.
import json
import os
import copy
import glob
import warnings
from tqdm import tqdm
from typing import Any, Dict
import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_tar

class PublicCaseDataset(InMemoryDataset):
    """
    PFDeltaDataset: PyTorch Geometric InMemoryDataset for PFDelta benchmark data.

    This class automatically downloads, extracts, and preprocesses data from the
    PFDelta benchmark on HuggingFace 
    (https://huggingface.co/datasets/pfdelta/pfdelta/tree/main). 
    
    It converts raw
    JSON power flow samples 
    into PyTorch Geometric `HeteroData` objects 
    for use
    with 
    graph-based learning models (e.g., CANOS, GNS, PFNet).

    The dataset supports multiple loading and processing modes:
        - Task-specific datasets: For supervised learning on benchmark tasks
          (e.g., 1.x, 2.x, 3.x, 4.x)
        - Analysis datasets: For per-case data inspection, visualization, and
          analytical performance studies

    Processed datasets are automatically organized by task, case, perturbation
    type, and feasibility regime within the downloaded dataset's directory structure.
    # 数据集的目录结构如下：
    # root_dir/
    # ├── case14/
    # │   ├── n/
    # │   │   ├── feasible/
    # │   │   ├── near_infeasible/
    # │   │   └── approaching_infeasible/
    # │   ├── n-1/
    # │   │   ├── feasible/
    # │   │   ├── near_infeasible/
    # │   │   └── approaching_infeasible/
    # │   └── n-2/
    # │       ├── feasible/
    # │       ├── near_infeasible/
    # │       └── approaching_infeasible/
    # ├── case30/
    # │   ├── n/
    # │   │   ├── feasible/
    # │   │   ├── near_infeasible/
    # │   │   └── approaching_infeasible/
    # │   ├── n-1/
    # │   │   ├── feasible/
    # │   │   ├── near_infeasible/
    # │   │   └── approaching_infeasible/
    # │   └── n-2/
    # │       ├── feasible/
    # │       ├── near_infeasible/
    # │       └── approaching_infeasible/
   

    #
    """

    def __init__(self, root_dir: str='data', case_name: str='case14', perturbation: str='n', feasibility_type: str='feasible', n_samples: int=-1, split: str='train', model: str='', task: Any=1.3, add_bus_type: bool=False, transform: Any=None, pre_transform: Any=None, pre_filter: Any=None, force_reload: bool=False):
        """
        Initialize the PFDeltaDataset class. 
        # 初始化数据集类，设置数据集的目录结构和元数据，
        # 括号里是函数的参数

        Parameters
        ----------
        root_dir : str
            Root data directory.
        case_name : str
            Power system case folder name (e.g., "case14").
        perturbation : str
            Grid contingency type ("n", "n-1", or "n-2").
        feasibility_type : str
            Feasibility regime for analytical mode 
            ("feasible", "near infeasible", or "approaching infeasible").
        n_samples : int
            Number of samples to load. 
            If < 0, loads all available samples.
        split : str
            Dataset split to load ("train", "val", "test", "all", or 
            "separate_{casename}_{split}_{feasibility}_{grid_type}").
            # 数据集的划分方式，可选值为"train"、"val"、"test"、"all"
            # 或"separate_{casename}_{split}_{feasibility}_{grid_type}"
            # 分别对应不同的数据集划分方式，比如"train"、"val"、"test"、"all"
            # 或"separate_case14_train_feasible_n"、"separate_case14_val_feasible_n"等
        model : str
            Model shorthand used for processed dataset naming.
            # 模型的简写名称，用于处理后的数据集命名，
            # 比如"case14_n_feasible_1.3"
        task : float or str
            Benchmark task ID (e.g., 1.3, 3.1, 4.2) or "analysis" for analytical mode.
        add_bus_type : bool
            Whether to include bus-type-specific node sets in HeteroData graphs.
            # 是否在HeteroData图中包含特定母线类型的节点集
        transform, pre_transform, pre_filter : callable, optional
            PyTorch Geometric dataset hooks for on-the-fly or preprocessing transforms.
            # PyTorch Geometric数据集的钩子函数，用于在加载数据时进行实时转换或预处理
        force_reload : bool
            If True, forces the dataset to reprocess even if cache exists.
            # 如果为True，则强制重新处理数据集，即使缓存存在也会重新处理
        Notes
        -----
        This initializer configures the dataset's directory structure and metadata.
        # 初始化数据集类，设置数据集的目录结构和元数据，
        # 括号里是函数的参数
        The raw-to-graph conversion occurs within the process() method, automatically
        triggered by the parent InMemoryDataset when required. 
        # 当缓存文件不存在时， InMemoryDataset 会自动调用 process 方法
        Also, the input for feasibility_type is only applicable when the task is on analysis mode. 
        # 当任务为分析模式时，feasibility_type参数才适用
        """
        self.split = split
        self.case_name = case_name
        self.force_reload = force_reload
        self.add_bus_type = add_bus_type
        self.task = task
        self.model = model
        self.root = os.path.join(root_dir, case_name)
        self.perturbation = perturbation
        self.feasibility_type = feasibility_type
        self.n_samples = n_samples
        if task in [3.2, 3.3]:
            self._custom_processed_dir = os.path.join(self.root, 'processed', f'combined_task_{task}_{model}_')
        elif task == 'analysis':
            self._custom_processed_dir = os.path.join(self.root, 'processed', f'task_{self.task}_{self.case_name}_{self.perturbation}_{self.feasibility_type}_{self.n_samples}')
            self.split = 'all'
        else:
            self._custom_processed_dir = os.path.join(self.root, 'processed', f'combined_task_{task}_{model}_{self.case_name}')
        self.all_case_names = ['case14', 'case30', 'case57', 'case118', 'case500', 'case2000']
        self.task_config = {1.1: {'feasible': {'n': 54000, 'n-1': 0, 'n-2': 0}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 1.2: {'feasible': {'n': 27000, 'n-1': 27000, 'n-2': 0}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 1.3: {'feasible': {'n': 18000, 'n-1': 18000, 'n-2': 18000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 2.1: {'feasible': {'n': 18000, 'n-1': 18000, 'n-2': 18000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 2.2: {'feasible': {'n': 12000, 'n-1': 12000, 'n-2': 12000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 2.3: {'feasible': {'n': 6000, 'n-1': 6000, 'n-2': 6000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 3.1: {'feasible': {'n': 18000, 'n-1': 18000, 'n-2': 18000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 3.2: {'feasible': {'n': 6000, 'n-1': 6000, 'n-2': 6000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 3.3: {'feasible': {'n': 6000, 'n-1': 6000, 'n-2': 6000}, 'near infeasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 4.1: {'near infeasible': {'n': 1800, 'n-1': 1800, 'n-2': 1800}, 'feasible': {'n': 16200, 'n-1': 16200, 'n-2': 16200}}, 4.2: {'approaching infeasible': {'n': 7200, 'n-1': 7200, 'n-2': 7200}, 'near infeasible': {'n': 1800, 'n-1': 1800, 'n-2': 1800}, 'feasible': {'n': 9000, 'n-1': 9000, 'n-2': 9000}}, 4.3: {'near infeasible': {'n': 3600, 'n-1': 3600, 'n-2': 3600}, 'approaching infeasible': {'n': 14400, 'n-1': 14400, 'n-2': 14400}, 'feasible': {'n': 0, 'n-1': 0, 'n-2': 0}}, 'analysis': {'feasible': {'n': 56000, 'n-1': 29000, 'n-2': 20000}, 'near infeasible': {'n': 2000, 'n-1': 2000, 'n-2': 2000}, 'approaching infeasible': {'n': 7200, 'n-1': 7200, 'n-2': 7200}}}
        self.test_config = {'feasible': {'n': 2000, 'n-1': 2000, 'n-2': 2000}, 'approaching infeasible': None, 'near infeasible': {'n': 200, 'n-1': 200, 'n-2': 200}}
        self.task_split_config = {3.1: {'train': [self.case_name], 'val': self.all_case_names, 'test': self.all_case_names}, 3.2: {'train': ['case14', 'case30', 'case57'], 'val': ['case118', 'case500', 'case2000'], 'test': ['case118', 'case500', 'case2000']}, 3.3: {'train': ['case118', 'case500', 'case2000'], 'val': ['case14', 'case30', 'case57'], 'test': ['case14', 'case30', 'case57']}}
        super().__init__(self.root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.split)

    def _split_to_idx(self):
        return {'train': 0, 'val': 1, 'test': 2}[self.split]

    @property
    def raw_file_names(self) -> list:
        if self.task in [3.1, 3.2, 3.3, 'analysis']:
            case_names = self.all_case_names
        else:
            case_names = [self.case_name]
        for case_name in case_names:
            case_raw_dir = os.path.join(self.root, case_name)
            if not os.path.exists(case_raw_dir):
                return []
        return sorted([os.path.join(self.task, self.casename, f) for f in os.listdir(case_raw_dir) if f.endswith('.json') and f.startswith(self.casename)])

    @property
    def processed_dir(self) -> str:
        return self._custom_processed_dir

    @property
    def processed_file_names(self) -> list:
        if self.task == 'analysis':
            return ['all.pt']
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        processed_dir = self.processed_dir
        required_files = self.processed_file_names
        if all((os.path.exists(os.path.join(processed_dir, f)) for f in required_files)):
            print('Processed data already exists. Skipping download.')
            return
        print(f'Downloading files for task {self.task}...')
        if self.task in [3.1, 3.2, 3.3]:
            case_names = self.all_case_names
        else:
            case_names = [self.case_name]
        base_url = 'https://huggingface.co/datasets/pfdelta/pfdelta/resolve/main'
        shuffle_download_path = self.root
        shuffle_files_dir = os.path.join(shuffle_download_path, 'shuffle_files')
        if os.path.exists(shuffle_files_dir):
            print('Shuffle files already exist. Skipping download.')
        else:
            print('Downloading shuffle files...')
            file_url = f'{base_url}/shuffle_files.tar.gz'
            shuffle_files_path = download_url(file_url, shuffle_download_path, log=True)
            extract_tar(shuffle_files_path, shuffle_download_path)
        for case_name in case_names:
            data_url = f'{base_url}/{case_name}.tar.gz'
            case_raw_dir = self.root
            os.makedirs(case_raw_dir, exist_ok=True)
            case_data_dir = self.root
            has_data = os.path.isdir(os.path.join(case_data_dir, 'n', 'raw')) or os.path.isdir(os.path.join(case_data_dir, 'raw'))
            if has_data:
                print(f'{case_name} data already exists at {case_data_dir}. Skipping download.')
                continue
            print(f'Downloading {case_name} data from {data_url} ...')
            try:
                download_path = case_raw_dir
                data_path = download_url(data_url, download_path, log=True)
                extract_tar(data_path, download_path)
                os.unlink(data_path)
                print(f'Extracted {case_name} data to {case_raw_dir}')
            except Exception as e:
                print(f'Failed to download or extract {case_name} data: {e}')

    def build_heterodata(self, pm_case: Dict[str, Any], is_cpf_sample: bool=False) -> HeteroData:
        """
        Convert a parsed PowerModels case dictionary into a HeteroData graph.

        This method parses a PowerModels.jl-style case JSON (produced by the PFDelta
        data generation pipeline) and converts it into a typed heterogeneous graph
        suitable for PyTorch Geometric. Each case contains detailed electrical
        network data, including buses, generators, loads, and branches, along with
        solved power flow variables.

        The resulting `HeteroData` object includes:
        - Node sets for buses, generators, and loads (and optionally PV, PQ, slack subsets).
        - Edge types linking physical entities (e.g., ("bus","branch","bus"), ("gen","gen_link","bus")).
        - Node- and edge-level attributes for network parameters, limits, and solved quantities.
        - Labels (.y) for supervised learning tasks (e.g., voltage angles and magnitudes).

        Parameters
        ----------
        pm_case : dict
            PowerModels network dictionary parsed from a JSON file.  
            - For continuation power flow (CPF) samples, this contains a "solved_net" entry 
            holding both network and solution data.  
            - For standard power flow samples, the dictionary includes "network" and 
            `"solution"` keys.

pm_case = {
    # 网络基本信息
    "name": "2-bus system",
    "baseMVA": 100.0,  # 基准功率 (MVA)
    "bus": {  # 母线数据
        1: {  # 母线 1
            "bus_i": 1,  # 母线编号
            "type": 3,  # 母线类型 (3: 平衡节点)
            "Pd": 0.0,  # 有功负荷 (MW)
            "Qd": 0.0,  # 无功负荷 (Mvar)
            "Gs": 0.0,  # 并联电导 (pu)
            "Bs": 0.0,  # 并联电纳 (pu)
            "area": 1,  # 区域编号
            "Vm": 1.0,  # 电压幅值 (pu)
            "Va": 0.0,  # 电压角度 (度)
            "baseKV": 138.0,  # 基准电压 (kV)
            "zone": 1,  # 分区编号
            "Vmax": 1.1,  # 最大电压 (pu)
            "Vmin": 0.9  # 最小电压 (pu)
        },
        2: {  # 母线 2
            "bus_i": 2,  # 母线编号
            "type": 1,  # 母线类型 (1: PQ节点)
            "Pd": 100.0,  # 有功负荷 (MW)
            "Qd": 50.0,  # 无功负荷 (Mvar)
            "Gs": 0.0,  # 并联电导 (pu)
            "Bs": 0.0,  # 并联电纳 (pu)
            "area": 1,  # 区域编号
            "Vm": 0.95,  # 电压幅值 (pu)
            "Va": -5.0,  # 电压角度 (度)
            "baseKV": 138.0,  # 基准电压 (kV)
            "zone": 1,  # 分区编号
            "Vmax": 1.1,  # 最大电压 (pu)
            "Vmin": 0.9  # 最小电压 (pu)
        }
    },
    "gen": {  # 发电机数据
        1: {  # 发电机 1 (连接到母线 1)
            "gen_bus": 1,  # 连接的母线编号
            "Pg": 100.0,  # 有功出力 (MW)
            "Qg": 50.0,  # 无功出力 (Mvar)
            "Qmax": 100.0,  # 最大无功出力 (Mvar)
            "Qmin": -50.0,  # 最小无功出力 (Mvar)
            "Vg": 1.0,  # 机端电压 (pu)
            "mBase": 100.0,  # 机端基准功率 (MVA)
            "status": 1,  # 状态 (1: 运行, 0: 停运)
            "Pmax": 200.0,  # 最大有功出力 (MW)
            "Pmin": 0.0,  # 最小有功出力 (MW)
            "Pc1": 0.0,  # 成本曲线参数 1
            "Pc2": 0.0,  # 成本曲线参数 2
            "Qc1min": 0.0,  # 无功成本曲线参数
            "Qc1max": 0.0,
            "Qc2min": 0.0,
            "Qc2max": 0.0,
            "ramp_agc": 0.0,
            "ramp_10": 0.0,
            "ramp_30": 0.0,
            "ramp_q": 0.0,
            "apf": 0.0
        }
    },
    "branch": {  # 线路数据
        1: {  # 线路 1 (连接母线 1 和 2)
            "f_bus": 1,  # 首端母线
            "t_bus": 2,  # 末端母线
            "br_r": 0.02,  # 线路电阻 (pu)
            "br_x": 0.1,  # 线路电抗 (pu)
            "br_b": 0.0,  # 线路电纳 (pu)
            "rate_a": 200.0,  # 长期额定容量 (MVA)
            "rate_b": 200.0,  # 短期额定容量 (MVA)
            "rate_c": 200.0,  # 紧急额定容量 (MVA)
            "ratio": 0.0,  # 变比 (非变压器为 0)
            "angle": 0.0,  # 相角 (非变压器为 0)
            "status": 1,  # 状态 (1: 运行, 0: 停运)
            "angmin": -360.0,  # 最小相角差 (度)
            "angmax": 360.0  # 最大相角差 (度)
        }
    },
    # 潮流计算解决方案
    "solution": {
        "bus": {  # 母线解决方案
            1: {
                "Vm": 1.0,  # 计算得到的电压幅值 (pu)
                "Va": 0.0,  # 计算得到的电压角度 (度)
                "net_p": 1.0,  # 净有功功率注入 (pu)
                "net_q": 0.5  # 净无功功率注入 (pu)
            },
            2: {
                "Vm": 0.95,  # 计算得到的电压幅值 (pu)
                "Va": -5.0,  # 计算得到的电压角度 (度)
                "net_p": -1.0,  # 净有功功率注入 (pu)
                "net_q": -0.5  # 净无功功率注入 (pu)
            }
        },
        "gen": {  # 发电机解决方案
            1: {
                "pg": 1.0,  # 有功出力 (pu)
                "qg": 0.5   # 无功出力 (pu)
            }
        },
        "branch": {  # 线路解决方案
            1: {
                "pf": 1.05,  # 首端有功功率 (pu)
                "qf": 0.52,  # 首端无功功率 (pu)
                "pt": -1.0,  # 末端有功功率 (pu)
                "qt": -0.5,  # 末端无功功率 (pu)
                "i_fr": 1.0,  # 首端电流 (pu)
                "i_to": 1.0,  # 末端电流 (pu)
                "losses": {"p": 0.05, "q": 0.02}  # 线路损耗
            }
        }
    }
}  

        is_cpf_sample : bool, optional (default=False)
            Whether the case is a CPF sample (i.e., derived from a near- or approaching-
            infeasible regime). Enables alternative handling for "solved_net" format.

        Returns
        -------
        torch_geometric.data.HeteroData
            A heterogeneous graph representation of the power network containing node and edge
            types with structured features, physical parameters, and solved variables.

        Notes
        -----
        - Node features include bus demands, generations, shunts, voltage magnitudes/angles, and limits.
        - Edge features include branch impedances, admittances, tap ratios, and flow limits.
        - Supports optional sub-node sets (PV, PQ, slack) and linking edges if 
        self.add_bus_type=True.
        - The resulting graphs are standardized for use across multiple PFDelta tasks (1.x–4.x).
        """
        data = HeteroData()
        if is_cpf_sample:
            network_data = pm_case['solved_net']
            solution_data = pm_case['solved_net']
        else:
            network_data = pm_case['network']
            solution_data = pm_case['solution']['solution']
        (pf_x, pf_y) = ([], [])
        bus_voltages = []
        bus_type = []
        bus_shunts = []
        (bus_gen, bus_demand) = ([], [])
        voltage_limits = []
        (PQ_bus_x, PQ_bus_y) = ([], [])
        (PV_bus_x, PV_bus_y) = ([], [])
        (PV_demand, PV_generation) = ([], [])
        (slack_x, slack_y) = ([], [])
        (slack_demand, slack_generation) = ([], [])
        (PV_to_bus, PQ_to_bus, slack_to_bus) = ([], [], [])
        (pq_idx, pv_idx, slack_idx) = (0, 0, 0)
        for (bus_id_str, bus) in sorted(network_data['bus'].items(), key=lambda x: int(x[0])):
            bus_id = int(bus_id_str)
            bus_idx = bus_id - 1
            bus_sol = solution_data['bus'][bus_id_str]
            (va, vm) = (bus_sol['va'], bus_sol['vm'])
            bus_voltages.append(torch.tensor([va, vm]))
            (vmin, vmax) = (bus['vmin'], bus['vmax'])
            voltage_limits.append(torch.tensor([vmin, vmax]))
            (gs, bs) = (0.0, 0.0)
            for shunt in network_data['shunt'].values():
                if int(shunt['shunt_bus']) == bus_id:
                    gs += shunt['gs']
                    bs += shunt['bs']
            bus_shunts.append(torch.tensor([gs, bs]))
            (pd, qd) = (0.0, 0.0)
            for load in network_data['load'].values():
                if int(load['load_bus']) == bus_id:
                    pd += load['pd']
                    qd += load['qd']
            bus_demand.append(torch.tensor([pd, qd]))
            (pg, qg) = (0.0, 0.0)
            for (gen_id, gen) in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
                if int(gen['gen_bus']) == bus_id:
                    if gen['gen_status'] == 1:
                        gen_sol = solution_data['gen'][gen_id]
                        pg += gen_sol['pg']
                        qg += gen_sol['qg']
                    elif is_cpf_sample:
                        assert gen['pg'] == 0 and gen['qg'] == 0, f'Expected gen {gen_id} to be off'
                    else:
                        assert solution_data['gen'].get(gen_id) is None, f'Expected gen {gen_id} to be off.'
            bus_gen.append(torch.tensor([pg, qg]))
            bus_type_now = bus['bus_type']
            if bus_type_now == 2 and pg == 0.0 and (qg == 0.0):
                bus_type_now = 1
                warnings.warn(f'Warning: Changed bus {bus_id} type from PV to PQ because it has no generation.')
            bus_type.append(torch.tensor(bus_type_now))
            if bus_type_now == 1:
                pf_x.append(torch.tensor([pd, qd]))
                pf_y.append(torch.tensor([va, vm]))
                PQ_bus_x.append(torch.tensor([pd, qd]))
                PQ_bus_y.append(torch.tensor([va, vm]))
                PQ_to_bus.append(torch.tensor([pq_idx, bus_idx]))
                pq_idx += 1
            elif bus_type_now == 2:
                pf_x.append(torch.tensor([pg - pd, vm]))
                pf_y.append(torch.tensor([qg - qd, va]))
                PV_bus_x.append(torch.tensor([pg - pd, vm]))
                PV_bus_y.append(torch.tensor([qg - qd, va]))
                PV_demand.append(torch.tensor([pd, qd]))
                PV_generation.append(torch.tensor([pg, qg]))
                PV_to_bus.append(torch.tensor([pv_idx, bus_idx]))
                pv_idx += 1
            elif bus_type_now == 3:
                pf_x.append(torch.tensor([va, vm]))
                pf_y.append(torch.tensor([pg - pd, qg - qd]))
                slack_x.append(torch.tensor([va, vm]))
                slack_y.append(torch.tensor([pg - pd, qg - qd]))
                slack_demand.append(torch.tensor([pd, qd]))
                slack_generation.append(torch.tensor([pg, qg]))
                slack_to_bus.append(torch.tensor([slack_idx, bus_idx]))
                slack_idx += 1
        (generation, limits, slack_gen) = ([], [], [])
        for (gen_id, gen) in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
            if gen['gen_status'] == 1:
                gen_sol = solution_data['gen'][gen_id]
                (pmin, pmax, qmin, qmax) = (gen['pmin'], gen['pmax'], gen['qmin'], gen['qmax'])
                (pgen, qgen) = (gen_sol['pg'], gen_sol['qg'])
                limits.append(torch.tensor([pmin, pmax, qmin, qmax]))
                generation.append(torch.tensor([pgen, qgen]))
                is_slack = torch.tensor(1 if network_data['bus'][str(gen['gen_bus'])]['bus_type'] == 3 else 0, dtype=torch.bool)
                slack_gen.append(is_slack)
            elif is_cpf_sample:
                assert solution_data['gen'][gen_id]['pg'] == 0 and solution_data['gen'][gen_id]['qg'] == 0, f'Expected gen {gen_id} to be off'
            else:
                assert solution_data['gen'].get(gen_id) is None, f'Expected gen {gen_id} to be off.'
        demand = []
        for (load_id, load) in sorted(network_data['load'].items(), key=lambda x: int(x[0])):
            (pd, qd) = (load['pd'], load['qd'])
            demand.append(torch.tensor([pd, qd]))
        (edge_index, edge_attr, edge_label, edge_limits) = ([], [], [], [])
        for (branch_id_str, branch) in sorted(network_data['branch'].items(), key=lambda x: int(x[0])):
            if branch['br_status'] == 0:
                continue
            from_bus = int(branch['f_bus']) - 1
            to_bus = int(branch['t_bus']) - 1
            edge_index.append(torch.tensor([from_bus, to_bus]))
            edge_attr.append(torch.tensor([branch['br_r'], branch['br_x'], branch['g_fr'], branch['b_fr'], branch['g_to'], branch['b_to'], branch['tap'], branch['shift']]))
            edge_limits.append(torch.tensor([branch['rate_a']]))
            branch_sol = solution_data['branch'].get(branch_id_str)
            assert branch_sol is not None, f'Missing solution for active branch {branch_id_str}'
            edge_label.append(torch.tensor([branch_sol['pf'], branch_sol['qf'], branch_sol['pt'], branch_sol['qt']]))
        gen_to_bus_index = []
        for (gen_id, gen) in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
            if gen['gen_status'] == 1:
                gen_bus = torch.tensor(gen['gen_bus']) - 1
                gen_to_bus_index.append(torch.tensor([int(gen_id) - 1, gen_bus]))
        load_to_bus_index = []
        for (load_id, load) in sorted(network_data['load'].items(), key=lambda x: int(x[0])):
            load_bus = torch.tensor(load['load_bus']) - 1
            load_to_bus_index.append(torch.tensor([int(load_id) - 1, load_bus]))
        data['bus'].x = torch.stack(pf_x)
        data['bus'].y = torch.stack(pf_y)
        data['bus'].bus_gen = torch.stack(bus_gen)
        data['bus'].bus_demand = torch.stack(bus_demand)
        data['bus'].bus_voltages = torch.stack(bus_voltages)
        data['bus'].bus_type = torch.stack(bus_type)
        data['bus'].shunt = torch.stack(bus_shunts)
        data['bus'].limits = torch.stack(voltage_limits)
        data['gen'].limits = torch.stack(limits)
        data['gen'].generation = torch.stack(generation)
        data['gen'].slack_gen = torch.stack(slack_gen)
        data['load'].demand = torch.stack(demand)
        if self.add_bus_type:
            data['PQ'].x = torch.stack(PQ_bus_x)
            data['PQ'].y = torch.stack(PQ_bus_y)
            data['PV'].x = torch.stack(PV_bus_x)
            data['PV'].y = torch.stack(PV_bus_y)
            data['PV'].generation = torch.stack(PV_generation)
            data['PV'].demand = torch.stack(PV_demand)
            data['slack'].x = torch.stack(slack_x)
            data['slack'].y = torch.stack(slack_y)
            data['slack'].generation = torch.stack(slack_generation)
            data['slack'].demand = torch.stack(slack_demand)
        for (link_name, edges) in {('bus', 'branch', 'bus'): edge_index, ('gen', 'gen_link', 'bus'): gen_to_bus_index, ('load', 'load_link', 'bus'): load_to_bus_index}.items():
            edge_tensor = torch.stack(edges, dim=1)
            data[link_name].edge_index = edge_tensor
            if link_name != ('bus', 'branch', 'bus'):
                data[link_name[2], link_name[1], link_name[0]].edge_index = edge_tensor.flip(0)
            if link_name == ('bus', 'branch', 'bus'):
                data[link_name].edge_attr = torch.stack(edge_attr)
                data[link_name].edge_label = torch.stack(edge_label)
                data[link_name].edge_limits = torch.stack(edge_limits)
        if self.add_bus_type:
            for (link_name, edges) in {('PV', 'PV_link', 'bus'): PV_to_bus, ('PQ', 'PQ_link', 'bus'): PQ_to_bus, ('slack', 'slack_link', 'bus'): slack_to_bus}.items():
                edge_tensor = torch.stack(edges, dim=1)
                data[link_name].edge_index = edge_tensor
                data[link_name[2], link_name[1], link_name[0]].edge_index = edge_tensor.flip(0)
        return data

    def get_analysis_data(self, case_root: str) -> list[HeteroData]:
        """
        Collect and convert raw JSON files for the 'analysis' task.

        This method gathers solved power flow samples from the appropriate 
        feasibility regime—feasible, near infeasible (nose), or approaching 
        infeasible (around_nose)—and converts each JSON case into a 
        `HeteroData` object suitable for graph-based analysis.
        # 上面这段话的意思是,将分析任务中的原始JSON文件收集并转换为适合图分析的HeteroData对象。
        The number of loaded samples is determined by self.n_samples or 
        the task configuration limits defined in self.task_config.
        # 上面这段话的意思是，根据任务配置和样本数量，确定要加载的样本数量。

        Parameters
        ----------
        case_root : str
            Path to the case directory containing subfolders such as 
            raw/, nose/, or around_nose/.

        Returns
        -------
        list of HeteroData
            List of converted power flow samples up to the configured dataset size.

        Notes
        -----
        - Samples are treated as continuation power flow (CPF) cases when 
        the feasibility type is not 'feasible'.
        - File discovery is based on fixed directory naming conventions 
        within each case folder.
        """
        dataset_size = self.n_samples if self.n_samples > 0 else self.task_config[self.task][self.feasibility_type][self.perturbation]
        data_list = []
        if self.feasibility_type == 'feasible':
            raw_fnames = glob.glob(os.path.join(case_root, self.perturbation, 'raw', '*.json'))
        elif self.feasibility_type == 'near infeasible':
            raw_fnames = glob.glob(os.path.join(case_root, self.perturbation, 'nose', 'train', '*.json'))
            raw_fnames.extend(glob.glob(os.path.join(case_root, self.perturbation, 'nose', 'test', '*.json')))
        elif self.feasibility_type == 'approaching infeasible':
            raw_fnames = glob.glob(os.path.join(case_root, self.perturbation, 'around_nose', 'train', '*.json'))
        data_list = []
        is_cpf_sample = True if self.feasibility_type != 'feasible' else False
        for file in raw_fnames[:dataset_size]:
            with open(file, 'r') as f:
                pm_case = json.load(f)
            data = self.build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
            data_list.append(data)
        return data_list

    def get_shuffle_file_path(self, grid_type: str, case_root: str) -> str:
        """
        Return the path to the shuffle mapping JSON file for a given grid type.

        This utility determines which shuffle mapping to use when building 
        processed train/validation/test splits. For large-scale systems such 
        as case2000, a dedicated shuffle file is selected to match 
        dataset naming conventions.

        Parameters
        ----------
        grid_type : str
            Grid size category (e.g., 'small', 'medium', 'large') 
            used to locate the correct shuffle file.
        case_root : str
            Path to the case directory, used to detect special naming rules.

        Returns
        -------
        str
            Absolute path to the corresponding shuffle JSON file.
        """
        if 'case2000' in case_root:
            return os.path.join(self.root, 'shuffle_files', grid_type, 'raw_shuffle_2000.json')
        else:
            return os.path.join(self.root, 'shuffle_files', grid_type, 'raw_shuffle.json')

    def shuffle_split_and_save_data(self, case_root: str, split: str, *, save_subsplits=True) -> Dict[str, list]:
        """Create and save processed train/val/test splits for a given case.

        This method constructs processed PyTorch Geometric datasets for one
        power flow case by reading shuffle mappings, loading raw JSON samples,
        and converting each case to a `HeteroData` object. It handles both
        feasible and infeasible regimes and organizes processed data into
        per-split .pt files under structured folders.

        The function also returns in-memory lists of all processed samples,
        enabling downstream concatenation across multiple cases (e.g. for
        cross-case training or evaluation).

        Parameters
        ----------
        case_root : str
            Path to the directory containing all subfolders for a given
            power flow case (e.g. case14/n/raw/, case14/n-1/nose/train/).

        Returns
        -------
        Dict[str, list]
            Dictionary mapping split names ("train", "val", "test")
            to lists of processed `HeteroData` objects.

        Notes
        -----
        - For feasible data, split boundaries are derived from shuffle maps
        located in shuffle_files/ and follow a 90/10 train/val partition.
        The test set uses the last test_size samples.
        - For infeasible and approaching infeasible regimes, samples are
        directly loaded from their respective nose and around_nose folders.
        - Processed datasets are saved under:
        <case_root>/<grid_type>/processed/task_<task>_<feasibility>_<model>/split.pt.
        - The returned data lists can be concatenated to build cross-case datasets.
        """
        (task, model) = (self.task, self.model)
        task_config = self.task_config[task]
        total_data_list = []
        current_case_name = os.path.basename(case_root)
        if current_case_name in ['case2000', 'case500']:
            task_config = copy.deepcopy(task_config)
            for (feasibility_key, feas_values) in task_config.items():
                for (feas_type, feas_num) in feas_values.items():
                    if current_case_name == 'case2000':
                        feas_values[feas_type] = feas_num // 2
                    if current_case_name == 'case500' and feas_num > 0:
                        avail_path = os.path.join(case_root, feas_type, 'raw')
                        if os.path.isdir(avail_path):
                            avail_count = len([f for f in os.listdir(avail_path) if f.endswith('.json')])
                            test_cfg_dict = self.test_config.get(feasibility_key, {})
                            test_reserve = test_cfg_dict.get(feas_type, 0) if test_cfg_dict else 0
                            max_train = avail_count - test_reserve
                            if feas_values[feas_type] > max_train:
                                print(f'  Capping {current_case_name} {feas_type} from {feas_values[feas_type]} to {max_train} (available={avail_count}, test_reserve={test_reserve})')
                                feas_values[feas_type] = max(0, max_train)
        for (feasibility, train_cfg_dict) in task_config.items():
            test_cfg = self.test_config[feasibility]
            for grid_type in ['n', 'n-1', 'n-2']:
                train_size = train_cfg_dict[grid_type]
                test_size = test_cfg.get(grid_type) if test_cfg else 0
                if current_case_name == 'case2000':
                    test_size = test_size // 2
                if train_size == 0 and test_size == 0:
                    continue
                if feasibility == 'feasible':
                    shuffle_path = self.get_shuffle_file_path(grid_type, case_root)
                    with open(shuffle_path, 'r') as f:
                        shuffle_dict = json.load(f)
                    shuffle_map = {int(k): int(v) for (k, v) in shuffle_dict.items()}
                    raw_path = os.path.join(case_root, grid_type, 'raw')
                    raw_fnames = [os.path.join(raw_path, f'sample_{i + 1}.json') for i in shuffle_map.keys()]
                    fnames_shuffled = [raw_fnames[shuffle_map[i]] for i in shuffle_map.keys()]
                    entire_size = len(fnames_shuffled)
                    assert train_size + test_size <= entire_size, 'Test samples overlap with train samples!'
                    split_dict = {'train': fnames_shuffled[:int(0.9 * train_size)], 'val': fnames_shuffled[int(0.9 * train_size):int(train_size)], 'test': fnames_shuffled[-int(test_size):]}
                else:
                    infeasibility_type = 'around_nose' if feasibility == 'approaching infeasible' else 'nose'
                    infeasible_train_path = os.path.join(case_root, grid_type, infeasibility_type, 'train')
                    infeasible_test_path = os.path.join(case_root, grid_type, infeasibility_type, 'test')
                    train_files = sorted([os.path.join(infeasible_train_path, f) for f in os.listdir(infeasible_train_path) if f.endswith('.json')])[:train_size]
                    if infeasibility_type == 'nose':
                        test_files = sorted([os.path.join(infeasible_test_path, f) for f in os.listdir(infeasible_test_path) if f.endswith('.json')])[:test_size]
                    else:
                        test_files = None
                    split_idx = int(0.9 * len(train_files))
                    split_dict = {'train': train_files[:split_idx], 'val': train_files[split_idx:], 'test': test_files}
                files = split_dict[split]
                if not files:
                    continue
                print(f'Processing split: {model} {task} ' + f'{current_case_name} {grid_type} {feasibility} ' + f'{split} ({len(files)} files)')
                current_data_list = []
                is_cpf_sample = feasibility != 'feasible'
                for fname in tqdm(files, desc=f'Building {split} data'):
                    with open(fname, 'r') as f:
                        pm_case = json.load(f)
                    data = self.build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
                    current_data_list.append(data)
                total_data_list.extend(current_data_list)
                if save_subsplits:
                    (data, slices) = self.collate(current_data_list)
                    processed_path = os.path.join(case_root, f'{grid_type}/processed/task_{task}_{feasibility}_{model}')
                    os.makedirs(processed_path, exist_ok=True)
                    torch.save((data, slices), os.path.join(processed_path, f'{split}.pt'))
        return total_data_list

    def process(self):
        """
        This method serves as the main processing routine invoked by
        `InMemoryDataset`. It organizes how raw PowerModels cases are
        converted into processed `HeteroData` objects, split into
        train/validation/test partitions, and stored under structured
        directories.

        For the 'analysis' task, all samples are processed into a single
        all.pt file. For other tasks, this function either:
        - Processes one case and generates per-split .pt files, or
        - Combines multiple cases (tasks 3.x) into consolidated datasets
        for each split, saving them under combined processed folders.

        Notes
        -----
        - For 'analysis' tasks, processed data is stored under:
        <root>/processed/task_<task>_<case>_<perturbation>_<feasibility>_<n_samples>/all.pt.
        - For combined multi-case tasks (3.x), split datasets are saved under:
        <root>/processed/combined_task_<task>_<model>_<casename>/<split>.pt.
        - Internally uses shuffle_split_and_save_data and get_analysis_data 
        to build per-case datasets before collation.
        """
        (task, model) = (self.task, self.model)
        casename = None
        combined_data_lists = {'train': [], 'val': [], 'test': []}
        if task in [3.1, 3.2, 3.3]:
            case_roots = [os.path.join(self.root, case_name) for case_name in self.all_case_names]
            casename = self.case_name if task == 3.1 else ''
        else:
            case_roots = [self.root]
            casename = self.case_name
        if task == 'analysis':
            print(f'Processing data for task {task}')
            task_data_list = self.get_analysis_data(self.root)
            (data, slices) = self.collate(task_data_list)
            processed_path = os.path.join(self.root, 'processed', f'task_{self.task}_{self.case_name}_{self.perturbation}_{self.feasibility_type}_{self.n_samples}')
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            torch.save((data, slices), os.path.join(processed_path, 'all.pt'))
            return
        for case_root in case_roots:
            print(f'Processing combined data for task {task}')
            for split in combined_data_lists.keys():
                if task in [3.1, 3.2, 3.3]:
                    case_name = os.path.basename(case_root)
                    if case_name not in self.task_split_config[task][split]:
                        continue
                data_list = self.shuffle_split_and_save_data(case_root, split)
                combined_data_lists[split].extend(data_list)
        for (split, data_list) in combined_data_lists.items():
            if data_list:
                print(f'Collating combined {split} data with {len(data_list)} samples')
                (combined_data, combined_slices) = self.collate(data_list)
                concat_path = os.path.join(self.root, f'processed/combined_task_{task}_{model}_{casename}')
                if not os.path.exists(concat_path):
                    os.makedirs(concat_path)
                torch.save((combined_data, combined_slices), os.path.join(concat_path, f'{split}.pt'))
                print(f'Saved combined {split} data with {len(data_list)} samples')

    def load(self, split: str):
        """
        Loads dataset for the specified split.

        Parameters
        ----------
        split : str
            The split to load. Can be one of:
                - 'train', 'val', 'test'
                - 'separate_{casename}_{split}_{feasibility}_{grid_type}' 
                (used for task 3.1 or 4.x separate per-case loading)

        Raises
        ------
        ValueError
            If the split string is malformed.
        FileNotFoundError
            If the processed file is not found at the computed path.

        Notes
        -----
        For example:
        split="separate_case57_train_feasible_n-1" loads:
        root/case57/n-1/processed/task_{task}_{feasibility}_{model}/train.pt.
        """
        if 'separate' in split:
            (_, casename_str, split_str, feasibility_str, grid_type_str) = split.split('_')
            if feasibility_str == 'near infeasible':
                processed_path = os.path.join(self.root, f'{casename_str}', f'{grid_type_str}', 'processed', f'task_{self.task}_{feasibility_str}_{self.model}', f'{split_str}.pt')
            else:
                processed_path = os.path.join(self.root, f'{casename_str}', f'{grid_type_str}', 'processed', f'task_{self.task}_{feasibility_str}_{self.model}', f'{split_str}.pt')
            print(f'Loading {split} dataset from {processed_path}')
            (self.data, self.slices) = torch.load(processed_path, weights_only=False)
        else:
            processed_path = os.path.join(self.processed_dir, f'{split}.pt')
            print(f'Loading {split} dataset from {processed_path}')
            (self.data, self.slices) = torch.load(processed_path, weights_only=False)
