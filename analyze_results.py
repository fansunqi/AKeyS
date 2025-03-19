import os
import json
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path


def replace_extension_with_png(file_path):
    path = Path(file_path)
    new_file_path = path.with_suffix('.png')
    return new_file_path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True, help='Input json filename')
args = parser.parse_args()

filepath = args.filepath
with open(filepath, 'r') as f:
    data = json.load(f)

# 统计每个step的总数和正确数
step_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
for item in data.values():
    step = item['get_ans_step']
    step_stats[step]['total'] += 1
    if item['answer'] == item['label']:
        step_stats[step]['correct'] += 1

# 计算正确率并排序
step_accuracy = {
    step: {
        'total': stats['total'],
        'correct': stats['correct'],
        'accuracy': stats['correct'] / stats['total'] * 100
    }
    for step, stats in step_stats.items()
}

# 在排序前添加检查和处理
step_accuracy = {step: acc for step, acc in step_accuracy.items() if step is not None}

# 然后再排序
sorted_steps = sorted(step_accuracy.items())

# 准备绘图数据
steps, stats = zip(*sorted_steps)
totals = [s['total'] for s in stats]
accuracies = [s['accuracy'] for s in stats]

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# fig.suptitle('Analysis of get_ans_step Distribution and Accuracy', fontsize=14, y=0.95)

# 绘制样本数量柱状图
bars1 = ax1.bar(range(len(totals)), totals)
ax1.set_xticks(range(len(steps)))
ax1.set_xticklabels(steps, rotation=45, ha='right')
ax1.set_title('Sample Count by Step')
ax1.set_ylabel('Number of Samples')
ax1.grid(True, alpha=0.3)

# 在柱子上添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

# 绘制正确率柱状图
bars2 = ax2.bar(range(len(accuracies)), accuracies)
ax2.set_xticks(range(len(steps)))
ax2.set_xticklabels(steps, rotation=45, ha='right')
ax2.set_title('Accuracy by Step')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

# 在柱子上添加正确率标签
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 显示图表
viz_file = replace_extension_with_png(filepath)
plt.savefig(viz_file)

# 打印详细统计信息
print("\nDetailed statistics:")
print(f"{'Step':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<8}")
print("-" * 40)
for step, stat in sorted_steps:
    print(f"{step:<15} {stat['total']:<8} {stat['correct']:<8} {stat['accuracy']:.1f}%")

# 计算总体统计
total_samples = sum(s['total'] for s in step_accuracy.values())
total_correct = sum(s['correct'] for s in step_accuracy.values())
overall_accuracy = total_correct / total_samples * 100

print("\nOverall statistics:")
print(f"Total samples: {total_samples}")
print(f"Total correct: {total_correct}")
print(f"Overall accuracy: {overall_accuracy:.1f}%")


def main():

    # 统计每个步骤的正确和总数
    step_correct = defaultdict(int)
    step_total = defaultdict(int)
    
    for video_info in data.values():
        step = video_info.get('get_ans_step', 'unknown')  # 使用 'unknown' 作为默认值
        step_total[step] += 1
        if video_info['corr'] == 1:
            step_correct[step] += 1

    # 计算每个步骤的准确率
    step_accuracy = {
        step: step_correct[step] / total 
        for step, total in step_total.items()
    }

    # 排序并打印结果
    sorted_steps = sorted(step_accuracy.items())
    
    print("\nAccuracy by step:")
    for step, acc in sorted_steps:
        print(f"{step}: {acc:.3f} ({step_correct[step]}/{step_total[step]})")

    # 打印总体统计
    total_correct = sum(step_correct.values())
    total_samples = sum(step_total.values())
    print(f"\nOverall accuracy: {total_correct/total_samples:.3f} ({total_correct}/{total_samples})")

if __name__ == "__main__":
    main()