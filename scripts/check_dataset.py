import os

def check_dataset(txt_file):
    missing_files = []
    base_dir = os.path.dirname(txt_file)

    with open(txt_file, 'r') as f:
        for line in f:
            path, _ = line.strip().split()
            full_path = os.path.join(base_dir, path)
            if not os.path.exists(full_path):
                missing_files.append(full_path)

    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for f in missing_files[:10]:  # 最多只印前10個
            print(f)
        if len(missing_files) > 10:
            print(f"... and {len(missing_files) - 10} more.")
    else:
        print("✅ All image files found.")

if __name__ == "__main__":
    # 替換為你的 txt 檔案路徑
    check_dataset("/home/MILS_HW1/data/mini_imagenet/train.txt")
    check_dataset("/home/MILS_HW1/data/mini_imagenet/val.txt")
