import os
from pathlib import Path

def locate_hf_path(repo_id="HuggingFaceVLA/libero"):
    folder_name = f"datasets--{repo_id.replace('/', '--')}"
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    repo_path = cache_base / folder_name
    
    if not repo_path.exists():
        print(f"❌ 未找到目录: {repo_path}")
        return

    snapshots_path = repo_path / "snapshots"
    hashes = sorted([d for d in snapshots_path.iterdir() if d.is_dir()], 
                    key=lambda x: x.stat().st_mtime, reverse=True)
    
    if hashes:
        target_data_path = hashes[0] / "data" / "chunk-000"
        target_data_path.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ 确定的上传路径：\n{target_data_path}")
    else:
        print("❌ snapshots 下没有哈希文件夹。")

locate_hf_path()
