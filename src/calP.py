import random
import numpy as np
import time  # 用于统计运行时间


def simulate_draws():
    """模拟一轮抽卡，直到抽到6次大奖（含所有规则）"""
    draw_counts = [0] * 6  # 记录每次大奖的累计抽卡数
    prize_count = 0
    total_draws = 0
    guaranteed_120_used = False  # 120抽保底仅一次
    bonus_prize_at = 240  # 额外赠送节点
    eighty_draw_counter = 0  # 80抽必中奖计数器
    no_prize_streak = 0  # 未中奖连抽计数（用于概率递增）
    base_prize_prob = 0.008  # 基础中奖概率（大奖+小奖）

    while prize_count < 6:
        total_draws += 1
        eighty_draw_counter += 1
        no_prize_streak += 1
        prize_obtained = False
        normal_prize = False

        # 1. 概率递增规则（65抽未中奖后触发）
        current_prize_prob = base_prize_prob
        if no_prize_streak > 65:
            current_prize_prob += (no_prize_streak - 65) * 0.05
            current_prize_prob = min(current_prize_prob, 1.0)

        # 2. 80抽必中奖规则
        if eighty_draw_counter >= 80:
            if random.random() < 0.5:
                prize_obtained = True
            else:
                normal_prize = True
            eighty_draw_counter = 0
            no_prize_streak = 0

        # 3. 120抽保底（仅一次）
        elif not guaranteed_120_used and total_draws == 120:
            prize_obtained = True
            guaranteed_120_used = True
            eighty_draw_counter = 0
            no_prize_streak = 0

        # 4. 概率递增后的正常抽卡
        elif not prize_obtained and not normal_prize:
            if random.random() < current_prize_prob:
                if random.random() < 0.5:
                    prize_obtained = True
                else:
                    normal_prize = True
                no_prize_streak = 0
                eighty_draw_counter = 0

        # 5. 额外赠送大奖（240/480...抽）
        if total_draws == bonus_prize_at:
            if prize_count < 6:
                draw_counts[prize_count] = total_draws
                prize_count += 1
            bonus_prize_at += 240

        # 6. 记录获得的大奖
        if prize_obtained and prize_count < 6:
            draw_counts[prize_count] = total_draws
            prize_count += 1
            guaranteed_120_used = True  # 提前抽到大奖，保底失效

    return draw_counts, total_draws  # 同时返回累计抽卡数和总抽卡数


def run_simulation(n_rounds):
    """运行n轮模拟并计算平均值（优化100万轮效率）"""
    # 使用numpy预分配数组提升效率
    results_array = np.zeros((n_rounds, 6), dtype=np.int32)
    total_draws_array = np.zeros(n_rounds, dtype=np.int32)  # 存储每轮总抽卡数

    for i in range(n_rounds):
        draw_counts, total_draws = simulate_draws()
        results_array[i] = draw_counts
        total_draws_array[i] = total_draws

        # 每10万轮打印进度
        if (i + 1) % 100000 == 0:
            print(f"已完成 {i + 1}/{n_rounds} 轮模拟...")

    averages = results_array.mean(axis=0)
    return averages, results_array, total_draws_array


# 设置模拟轮数（100万次）
n_rounds = 10000000
start_time = time.time()

# 运行模拟
print(f"开始{n_rounds}轮抽卡模拟...")
averages, results_array, total_draws_array = run_simulation(n_rounds)

# 计算耗时
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n模拟完成！总耗时：{elapsed_time:.2f}秒（{elapsed_time / 60:.2f}分钟）")

# 输出结果
print("\n" + "-" * 80)
print(f"模拟轮数：{n_rounds}轮")
print("规则说明：")
print("- 65抽未中奖后，每抽中奖概率+5%（基础0.8%）")
print("- 每80抽内必定中奖，50%几率中大奖，50%中普通奖")
print("- 第120抽保底大奖（仅一次机会，提前抽到则失效）")
print("- 第240抽、第480抽...额外赠送一个大奖（不算抽取次数）")
print("-" * 80)

for i in range(6):
    print(f"抽到第{i + 1}次大奖平均累计抽卡数：{averages[i]:.2f}")

print("-" * 80)
# 总抽卡数统计（第6次大奖的累计抽卡数）
print(f"抽到6次大奖的最小总抽卡数：{np.min(total_draws_array)}")
print(f"抽到6次大奖的最大总抽卡数：{np.max(total_draws_array)}")
print(f"抽到6次大奖的中位数总抽卡数：{np.median(total_draws_array):.2f}")
print(f"抽到6次大奖的平均总抽卡数：{np.mean(total_draws_array):.2f}")
print(f"抽到6次大奖的标准差：{np.std(total_draws_array):.2f}")

# 新增：统计1200抽相关概率
over_1200_count = np.sum(total_draws_array > 1200)
over_1200_prob = over_1200_count / n_rounds * 100
exact_1200_count = np.sum(total_draws_array == 1200)
exact_1200_prob = exact_1200_count / n_rounds * 100
under_1200_count = np.sum(total_draws_array < 1200)
under_1200_prob = under_1200_count / n_rounds * 100

print("\n1200抽相关概率统计：")
print(f"恰好1200抽抽到6次大奖的概率：{exact_1200_prob:.6f}%")
print(f"超过1200抽才抽到6次大奖的概率：{over_1200_prob:.6f}%")
print(f"不到1200抽就抽到6次大奖的概率：{under_1200_prob:.2f}%")

# 统计120抽保底触发情况
guaranteed_triggered = np.sum(np.any(results_array == 120, axis=1))
print(f"\n120抽保底触发次数：{guaranteed_triggered}次（比例：{guaranteed_triggered / n_rounds * 100:.2f}%）")

# 统计额外赠送触发情况
bonus_240_count = np.sum(results_array[:, 3] == 240)  # 第4次大奖在240抽（额外赠送）
bonus_480_count = np.sum(results_array[:, 5] == 480)  # 第6次大奖在480抽（额外赠送）
print(f"240抽额外赠送触发比例：{bonus_240_count / n_rounds * 100:.2f}%")
print(f"480抽额外赠送触发比例：{bonus_480_count / n_rounds * 100:.2f}%")