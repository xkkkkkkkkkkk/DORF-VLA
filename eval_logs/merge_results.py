import json
from pathlib import Path

# 刚才跑的三个输出目录
log_dirs = [
    "eval_logs/libero_goal_part1",
    "eval_logs/libero_goal_part2",
    "eval_logs/libero_goal_part3"
]

combined_tasks = {}
total_success = 0.0
total_episodes = 0

for d in log_dirs:
    json_path = Path(d) / "eval_info.json" # 确保这里是实际生成的 json 名字
    if not json_path.exists():
        print(f"⚠️ 还没跑完或找不到文件: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 提取每个子任务的成绩
    if 'tasks' in data:
        for task_name, task_info in data['tasks'].items():
            combined_tasks[task_name] = task_info
            # 累加总成功次数和总回合数用于计算总平均分
            total_success += task_info['success_rate'] * task_info['n_episodes']
            total_episodes += task_info['n_episodes']

# 打印最终写进小论文的数据
if total_episodes > 0:
    final_avg_success = total_success / total_episodes
    print("="*50)
    print("🏆 LIBERO-Goal 最终合并成绩单 🏆")
    print("="*50)
    for task_name, task_info in combined_tasks.items():
        print(f" - {task_name}: {task_info['success_rate']*100:.1f}%")
    print("-" * 50)
    print(f"🌟 总平均成功率 (Avg Success Rate): {final_avg_success*100:.2f}%")
    print("="*50)
    print("💡 请将上方【总平均成功率】填入你的小论文 Baseline 表格中。")