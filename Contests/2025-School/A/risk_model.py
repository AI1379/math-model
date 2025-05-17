import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Literal, Union


class RiskModel:
    def __init__(
        self,
        nodes_data: pd.DataFrame,
        lines_data: pd.DataFrame,
        substation_data: Dict[str, int] = {},
        tie_switches: Dict[str, Tuple[int, int]] = {},
        dg_data: List[Dict[Literal["node", "capacity"], Union[int, float]]] = [],
        node_type: Dict[
            int, Literal["Residential", "Commercial", "Governmental", "Industrial"]
        ] = {},
        feeder_capacity: float = 2200,
        feeder_current_limit: float = 220,
        voltage: float = 10,
        dg_failure_rate: float = 0.005,
        load_failure_rate: float = 0.005,
        switch_failure_rate: float = 0.002,
        line_failure_rate_per_km: float = 0.002,
        **kwargs,
    ):
        """
        初始化风险模型

        :param nodes_data: 包含节点信息的 DataFrame
        :param lines_data: 包含线路拓扑信息的 DataFrame
        :param substation_data: 字典，包含变电站及其对应的节点，作为每个馈线的根节点
        :param tie_switches: 字典，包含联络开关及其对应的节点对
        :param dg_data: 分布式发电机数据，包含节点和容量
        :param feeder_capacity: 每个馈线的容量
        :param feeder_current_limit: 每个馈线的电流限制
        :param voltage: 系统电压，单位为 kV
        :param dg_failure_rate: 分布式发电机的故障率
        :param load_failure_rate: 负荷的故障率
        :param switch_failure_rate: 开关的故障率
        :param line_failure_rate_per_km: 每公里线路的故障率
        """
        self.node_weight = {
            "Residential": 0.2,
            "Commercial": 0.5,
            "Governmental": 0.8,
            "Industrial": 1.0,
        }
        self.node_type = node_type

        self.nodes_data = nodes_data
        self.lines_data = lines_data

        self.node_loads: Dict[int, float] = {}
        for i, row in nodes_data.iterrows():
            self.node_loads[i + 1] = row["有功P/kW"]

        self.dg_data = dg_data

        # 系统参数
        self.feeder_capacity = feeder_capacity
        self.feeder_current_limit = feeder_current_limit
        self.voltage = voltage

        # 故障率
        self.dg_failure_rate = dg_failure_rate
        self.load_failure_rate = load_failure_rate
        self.switch_failure_rate = switch_failure_rate
        self.line_failure_rate_per_km = line_failure_rate_per_km

        # 系统拓扑信息
        self.substation_data = substation_data
        self.tie_switches = tie_switches

        # 建图
        self.network = self._build_network()

        # 预处理馈线信息
        self.feeder_regions: Dict[str, List[int]] = {}
        self.feeder_table: Dict[int, str] = {}
        for substation, node in substation_data.items():
            dfs_result = sorted(list(nx.dfs_postorder_nodes(self.network, node)))
            self.feeder_regions[substation] = dfs_result
            for node in dfs_result:
                self.feeder_table[node] = substation

    def _build_network(self) -> nx.Graph:
        """
        构建网络拓扑图

        :return: 网络拓扑图
        """
        network = nx.Graph()
        for i in range(1, len(self.nodes_data) + 1):
            network.add_node(
                i,
                load=self.nodes_data.loc[i - 1, "有功P/kW"],
                is_dg=i in [dg["node"] for dg in self.dg_data],
            )

        # 添加边
        for _, row in self.lines_data.iterrows():
            start = int(row["起点"])
            end = int(row["终点"])
            length = float(row["长度/km"])
            resistance = float(row["电阻/Ω"])
            reactance = float(row["电抗/Ω"])

            network.add_edge(
                start,
                end,
                length=length,
                resistance=resistance,
                reactance=reactance,
                failure_rate=length * self.line_failure_rate_per_km,
            )

        return network

    def _calculate_remaining_capacity(self, feeder: str) -> float:
        """
        计算馈线剩余容量

        :param feeder: 馈线名称
        :return: 剩余容量
        """
        nodes = self.feeder_regions[feeder]
        total_load = sum(self.node_loads[node] for node in nodes)

        # 计算分布式发电机的贡献
        dg_contribution = sum(
            dg["capacity"] for dg in self.dg_data if dg["node"] in nodes
        )

        net_load = max(0, total_load - dg_contribution)

        return self.feeder_capacity - net_load

    def _calculate_node_failure_load_loss(self, node: int, is_dg: bool) -> float:
        """
        计算节点故障导致的负荷损失

        :param node: 节点编号
        :return: 负荷损失
        """
        # 如果是 dg 直接相连的节点，则认为本身不会发生故障，发生故障一定是 dg 故障
        if is_dg:
            return next(
                (dg["capacity"] for dg in self.dg_data if dg["node"] == node), 0
            )

        # 否则考虑节点故障导致的损失
        loss = max(
            self.node_loads[node]
            - self._calculate_remaining_capacity(self.feeder_table[node]),
            0,
        )

        return loss * self.node_weight[self.node_type[node]]

    def _calculate_line_failure_load_loss(self, line: Tuple[int, int]) -> float:
        """
        计算线路故障导致的负荷损失

        :param line: 线路两端节点的元组
        :return: 负荷损失
        """
        if self.feeder_table[line[0]] != self.feeder_table[line[1]]:
            return 0

        feeder = self.feeder_table[line[0]]

        temp_network = self.network.copy()
        temp_network.remove_edge(*line)

        # 计算断开后，馈线中哪些节点无法到达根节点
        affected_nodes = set()
        root_node = self.substation_data[feeder]
        for node in self.feeder_regions[feeder]:
            if node != root_node and not nx.has_path(temp_network, root_node, node):
                affected_nodes.add(node)

        # 计算负荷损失
        total_loss = sum(self.node_loads[node] for node in affected_nodes)

        # 减去断开后的子图中分布式发电机的贡献
        for dg in self.dg_data:
            if dg["node"] in affected_nodes:
                total_loss -= min(dg["capacity"], total_loss)

        # 计算联络线从其他馈线中转移的负荷
        for _, (node1, node2) in self.tie_switches.items():
            other_feeder = None
            if node1 in affected_nodes and node2 not in affected_nodes:
                other_feeder = self.feeder_table[node2]
            elif node2 in affected_nodes and node1 not in affected_nodes:
                other_feeder = self.feeder_table[node1]

            if other_feeder and other_feeder != feeder:
                other_remaining_capacity = self._calculate_remaining_capacity(
                    other_feeder
                )
                total_loss -= min(other_remaining_capacity, total_loss)

        # 计算平均权重，简化分析
        mean_weight = sum(
            self.node_weight[self.node_type[node]] for node in affected_nodes
        ) / len(affected_nodes)

        return total_loss * mean_weight

    def _calculate_switch_failure_load_loss(self, switch: Tuple[int, int]) -> float:
        """
        计算开关故障导致的负荷损失

        :param switch: 开关两端节点的元组
        :return: 负荷损失
        """
        # 简化处理，认为开关故障不导致负荷损失
        return 0

    def calculate_load_loss_risk(self) -> float:
        """
        计算负荷损失风险

        :return: 负荷损失风险
        """
        total_risk = 0.0

        # 分三部分计算风险
        # 然后累加到总风险中

        for node, data in self.network.nodes(data=True):
            if data["is_dg"]:
                failure_rate = self.dg_failure_rate
            else:
                failure_rate = self.load_failure_rate

            load_loss = self._calculate_node_failure_load_loss(node, data["is_dg"])

            total_risk += load_loss * failure_rate

        for u, v, data in self.network.edges(data=True):
            line_failure_rate = data["failure_rate"]
            load_loss = self._calculate_line_failure_load_loss((u, v))
            total_risk += load_loss * line_failure_rate

        for switch in self.tie_switches.values():
            failure_rate = self.switch_failure_rate
            load_loss = self._calculate_switch_failure_load_loss(switch)
            total_risk += load_loss * failure_rate

        return total_risk

    def calculate_overload_risk(self) -> float:
        """
        计算过载风险

        :return: 过载风险
        """
        # TODO: 分用户类型计算损失
        # TODO: 计算原件故障导致线路断路无法均衡负载导致的过负荷风险
        total_risk = 0.0
        for feeder in self.feeder_regions.keys():
            feeder_risk = self._calculate_feeder_overload_risk(feeder)
            total_risk += feeder_risk

        return max(0.1, total_risk)

    def _calculate_feeder_current(self, feeder: str) -> float:
        """
        计算馈线电流

        :param feeder: 馈线名称
        :return: 馈线电流
        """
        nodes = self.feeder_regions[feeder]

        total_load = sum(self.node_loads[node] for node in nodes)
        dg_capacity = sum(dg["capacity"] for dg in self.dg_data if dg["node"] in nodes)

        net_load = max(0, total_load - dg_capacity)  # 当前网络净负载
        excess_load = 0  # 额外负载

        if net_load > 0:
            current = net_load / (np.sqrt(3) * self.voltage)  # 考虑三相电流
        else:
            # DG的贡献大于负荷，考虑相邻馈线的负荷转移
            excess_load = dg_capacity - total_load

            for _, (node1, node2) in self.tie_switches.items():
                if node1 in nodes or node2 in nodes:
                    other_feeder = (
                        self.feeder_table[node2]
                        if node1 in nodes
                        else self.feeder_table[node1]
                    )
                    other_remaining_capacity = self._calculate_remaining_capacity(
                        other_feeder
                    )

                    transferable_load = min(excess_load, other_remaining_capacity)
                    excess_load -= transferable_load

            net_load = total_load + max(0, excess_load)
            current = net_load / (np.sqrt(3) * self.voltage)

        return current

    def _calculate_feeder_overload_risk(self, feeder: str) -> float:
        """
        计算馈线过载风险

        :param feeder: 馈线名称
        :return: 馈线过载风险
        """
        total_risk = 0.0
        current = self._calculate_feeder_current(feeder)

        # TODO: 认为馈线电流为正态分布
        mu = current
        sigma = 0.1 * current  # 假设标准差为平均值的10%
        mean_weight = sum(
            self.node_weight[self.node_type[node]]
            for node in self.feeder_regions[feeder]
        ) / len(self.feeder_regions[feeder])

        def overload_consequence(x: float) -> float:
            if x > self.feeder_current_limit * 1.1:
                return (
                    2
                    * np.sqrt(3)
                    * self.voltage
                    * (x - self.feeder_current_limit * 1.1 + 0.1)
                    * mean_weight
                )
            else:
                return (
                    0.2
                    * np.sqrt(3)
                    * self.voltage
                    * (x / (self.feeder_current_limit * 1.1))
                    * mean_weight
                )

        total_risk = quad(
            lambda x: overload_consequence(x) * norm.pdf(x, mu, sigma),
            0,
            np.inf,
        )[0]

        return total_risk

    def calculate_system_risk(self) -> Tuple[float, float, float]:
        """
        计算系统风险

        :return: 系统风险、负荷损失风险、过载风险
        """
        load_loss_risk = self.calculate_load_loss_risk()
        overload_risk = self.calculate_overload_risk()

        # 计算系统风险
        system_risk = load_loss_risk + overload_risk

        return system_risk, load_loss_risk, overload_risk

    def draw_network(self, path: Optional[str] = None) -> None:
        """
        绘制网络拓扑图

        :param path: 保存路径，默认为 None，表示直接显示图形
        """
        # TODO: 分类染色节点

        node_labels = {
            node: f"{node}\n{data['load']}kW"
            for node, data in self.network.nodes(data=True)
        }
        edge_labels = {
            (u, v): f"{data['length']:.2f} km"
            for u, v, data in self.network.edges(data=True)
        }

        for u, v in self.network.edges():
            self.network.edges[u, v]["weight"] = 1 / len(edge_labels[(u, v)])

        plt.figure(figsize=(14, 10))

        pos = nx.kamada_kawai_layout(self.network, weight="weight", scale=10)
        nx.draw(
            self.network,
            pos,
            node_size=700,
            node_color="lightblue",
        )

        nx.draw_networkx_labels(
            self.network,
            pos,
            labels=node_labels,
            font_size=8,
        )
        nx.draw_networkx_edge_labels(
            self.network,
            pos,
            edge_labels=edge_labels,
            font_size=6,
        )

        plt.title("Network Topology")
        plt.axis("off")

        if path:
            plt.savefig(path)
        else:
            plt.show()


def get_default_model() -> RiskModel:
    node_data = pd.read_excel("appendix.xlsx", sheet_name=0)
    line_data = pd.read_excel("appendix.xlsx", sheet_name=1)

    dg_data = [
        {"node": 13, "capacity": 300},
        {"node": 18, "capacity": 300},
        {"node": 22, "capacity": 300},
        {"node": 29, "capacity": 300},
        {"node": 32, "capacity": 300},
        {"node": 39, "capacity": 300},
        {"node": 48, "capacity": 300},
        {"node": 59, "capacity": 300},
    ]

    node_type = {
        1: "Residential",
        2: "Residential",
        3: "Residential",
        4: "Residential",
        5: "Residential",
        6: "Residential",
        7: "Industrial",
        8: "Residential",
        9: "Governmental",
        10: "Residential",
        11: "Commercial",
        12: "Industrial",
        13: "Residential",
        14: "Industrial",
        15: "Residential",
        16: "Commercial",
        17: "Residential",
        18: "Industrial",
        19: "Residential",
        20: "Residential",
        21: "Governmental",
        22: "Residential",
        23: "Residential",
        24: "Residential",
        25: "Residential",
        26: "Residential",
        27: "Commercial",
        28: "Residential",
        29: "Governmental",
        30: "Residential",
        31: "Commercial",
        32: "Industrial",
        33: "Commercial",
        34: "Commercial",
        35: "Residential",
        36: "Governmental",
        37: "Residential",
        38: "Commercial",
        39: "Residential",
        40: "Industrial",
        41: "Residential",
        42: "Commercial",
        43: "Residential",
        44: "Residential",
        45: "Governmental",
        46: "Residential",
        47: "Industrial",
        48: "Residential",
        49: "Commercial",
        50: "Residential",
        51: "Residential",
        52: "Residential",
        53: "Commercial",
        54: "Residential",
        55: "Residential",
        56: "Commercial",
        57: "Residential",
        58: "Governmental",
        59: "Residential",
        60: "Residential",
        61: "Governmental",
        62: "Residential",
    }

    tie_switches = {
        "S13-1": (13, 23),
        "S29-2": (29, 43),
        "S62-3": (62, 1),
    }

    substation_switches = {
        "CB1": 1,
        "CB2": 23,
        "CB3": 43,
    }

    model = RiskModel(
        nodes_data=node_data,
        lines_data=line_data,
        substation_data=substation_switches,
        tie_switches=tie_switches,
        dg_data=dg_data,
        node_type=node_type,
    )

    return model
