import torch
import os
# 导入你训练脚本里初始化 policy 的相关依赖
# from lerobot.policies.smolvla... import ...

def main():
    checkpoint_path = "dorf_vla_checkpoint_latest.pt" # 你的断点文件路径
    export_dir = "./my_finetuned_smolvla_libero"      # 你想要导出的标准 HF 文件夹名称

    print(f"📦 正在加载断点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 1. 像你训练代码里一样，初始化一个基础的空模型（或者加载你最初的基线权重）
    # policy = ... (这里抄一段你 train_dorf.py 里初始化 policy 的代码)
    
    # 2. 把断点里训练好的 VLA 权重强行注入进去
    print("💉 正在注入微调后的权重...")
    policy.load_state_dict(checkpoint['policy'])

    # 3. 使用 LeRobot/Hugging Face 原生的 save_pretrained 方法导出！
    print(f"💾 正在导出为标准 Hugging Face 格式至: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)
    policy.save_pretrained(export_dir)
    print("✅ 导出完成！现在你可以用官方脚本评测它了！")

if __name__ == "__main__":
    main()