import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, IO, Tuple
from risk_model import get_default_model


def analyze_dg_capacity(
    path: Optional[str] = None,
    output: Optional[IO[str]] = None,
    capacity_range: Tuple[float, float, float] = (1.0, 3.1, 0.3),
) -> None:
    """
    分析分布式发电容量对系统风险的影响

    :param path: 可选的路径参数，默认为None
    """
    model = get_default_model()

    dg_capacity_factors = np.arange(*capacity_range)

    results = {
        "capacity_factor": [],
        "total_dg_capacity": [],
        "load_loss_risk": [],
        "overload_risk": [],
        "system_risk": [],
    }

    base_dg_data = model.dg_data.copy()

    print("====== 分布式发电容量对系统风险的影响 ======", file=output)

    for factor in dg_capacity_factors:
        print(f"计算容量因子: {factor:.2f}", file=output)

        # 更新DG容量
        model.dg_data = list(
            map(
                lambda dg: {
                    "node": dg["node"],
                    "capacity": dg["capacity"] * factor,
                },
                base_dg_data,
            )
        )

        # 计算系统风险
        system_risk, load_loss_risk, overload_risk = model.calculate_system_risk()

        # 记录结果
        results["capacity_factor"].append(factor)
        results["total_dg_capacity"].append(sum(dg["capacity"] for dg in model.dg_data))
        results["load_loss_risk"].append(load_loss_risk)
        results["overload_risk"].append(overload_risk)
        results["system_risk"].append(system_risk)

        print(f"  失负荷风险: {load_loss_risk:.2f}", file=output)
        print(f"  过负荷风险: {overload_risk:.2f}", file=output)
        print(f"  系统总风险: {system_risk:.2f}", file=output)

    # 绘制图像
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 显示负号
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["capacity_factor"],
        results["load_loss_risk"],
        label="失负荷风险",
        marker="o",
    )
    plt.plot(
        results["capacity_factor"],
        results["overload_risk"],
        label="过负荷风险",
        marker="o",
    )
    plt.plot(
        results["capacity_factor"],
        results["system_risk"],
        label="系统总风险",
        marker="o",
    )
    plt.title("分布式发电容量对系统风险的影响")
    plt.xlabel("DG容量因子")
    plt.ylabel("风险值")

    if path:
        plt.savefig(path)

    print("保存图像到:", path, file=output)

    print("====== 馈线风险分析 ======", file=output)

    total_risk = (
        model.network.number_of_nodes() * model.load_failure_rate
        + model.network.size("failure_rate")
    )
    risk_ratios = [
        (
            model.network.subgraph(nodes).number_of_nodes() * model.load_failure_rate
            + model.network.subgraph(nodes).size("failure_rate")
        )
        / total_risk
        for feeder, nodes in model.feeder_regions.items()
    ]
    feeder_results = {
        feeder_name: [risk * ratio for risk in results["system_risk"]]
        for feeder_name, ratio in zip(model.feeder_regions.keys(), risk_ratios)
    }

    for idx, feeder_name in enumerate(model.feeder_regions.keys()):
        print(f"馈线 {feeder_name} 风险占比: {risk_ratios[idx]:.2%}", file=output)

    # 绘制馈线风险图
    plt.figure(figsize=(10, 6))
    for feeder_name, risks in feeder_results.items():
        plt.plot(
            results["capacity_factor"],
            risks,
            label=f"{feeder_name}风险",
            marker="o",
        )
    plt.title("馈线风险随DG容量因子变化")
    plt.xlabel("DG容量因子")
    plt.ylabel("风险值")
    plt.legend()
    if path:
        plt.savefig(path.replace(".png", "_feeder.png"))
    print(f"保存馈线风险图像到: {path.replace(".png", "_feeder.png")}", file=output)

    print(f"====== 分析结论 ======", file=output)
    system_risks = zip(
        dg_capacity_factors,
        results["system_risk"],
        results["load_loss_risk"],
        results["overload_risk"],
    )
    min_risk = min(system_risks, key=lambda x: x[1])
    print(f"最小系统风险发生在容量因子: {min_risk[0]:.2f}", file=output)
    print(f"最小系统风险: {min_risk[1]:.2f}", file=output)
    print(f"  对应失负荷风险: {min_risk[2]:.2f}", file=output)
    print(f"  对应过负荷风险: {min_risk[3]:.2f}", file=output)

    if not path:
        plt.show()


if __name__ == "__main__":
    analyze_dg_capacity()
