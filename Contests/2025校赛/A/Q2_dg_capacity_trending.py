import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from Q1_risk_model import get_default_model


def analyze_dg_capacity(path: Optional[str] = None) -> None:
    """
    分析分布式发电容量对系统风险的影响

    :param path: 可选的路径参数，默认为None
    """
    model = get_default_model()

    dg_capacity_factors = np.arange(1.0, 3.1, 0.1)

    results = {
        "capacity_factor": [],
        "total_dg_capacity": [],
        "load_loss_risk": [],
        "overload_risk": [],
        "system_risk": [],
    }

    base_dg_data = model.dg_data.copy()

    for factor in dg_capacity_factors:
        print(f"计算容量因子: {factor}")

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

        print(f"  失负荷风险: {load_loss_risk:.2f}")
        print(f"  过负荷风险: {overload_risk:.2f}")
        print(f"  系统总风险: {system_risk:.2f}")
        
    # TODO: 不同馈线的风险计算

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
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    analyze_dg_capacity()
