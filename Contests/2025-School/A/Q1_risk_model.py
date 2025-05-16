from typing import IO, Optional
from risk_model import get_default_model

def analyze_system_risk(
    topology_graph: Optional[str] = None, log_out: Optional[IO[str]] = None
) -> None:
    """
    分析系统风险
    """
    model = get_default_model()
    system_risk, load_loss_risk, overload_risk = model.calculate_system_risk()
    risk_level = "高" if system_risk > 30 else "中" if system_risk > 15 else "低"
    print(f"====== 系统风险分析 ======", file=log_out)
    print(f"系统风险: {system_risk}", file=log_out)
    print(f"失负荷风险: {load_loss_risk}", file=log_out)
    print(f"过负荷风险: {overload_risk}", file=log_out)
    print(f"====== 风险占比 ======", file=log_out)
    print(f"失负荷风险占比: {load_loss_risk / system_risk:.2%}", file=log_out)
    print(f"过负荷风险占比: {overload_risk / system_risk:.2%}", file=log_out)
    print(f"风险等级: {risk_level}", file=log_out)
    print(f"====== 网络拓扑图 ======", file=log_out)
    print(f"输出到: {topology_graph}", file=log_out)
    model.draw_network(topology_graph)


if __name__ == "__main__":
    analyze_system_risk()
