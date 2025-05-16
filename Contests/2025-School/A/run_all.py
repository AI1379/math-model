from Q1_risk_model import analyze_system_risk
from Q2_dg_capacity_trending import analyze_dg_capacity
import os

if __name__ == "__main__":
    os.makedirs("problem1", exist_ok=True)
    os.makedirs("problem2", exist_ok=True)
    
    topology_graph = "problem1/topology_graph.png"
    system_risk_log = "problem1/system_risk_log.txt"
    analyze_system_risk(
        topology_graph=topology_graph,
        log_out=open(system_risk_log, "w", encoding="utf-8"),
    )
    print(f"系统风险分析结果已保存到: {system_risk_log}")

    trending_graph = "problem2/dg_capacity_trending.png"
    dg_capacity_log = "problem2/dg_capacity_log.txt"
    analyze_dg_capacity(
        path=trending_graph,
        output=open(dg_capacity_log, "w", encoding="utf-8"),
    )
    print(f"分布式发电容量对系统风险的影响分析结果已保存到: {dg_capacity_log}")
