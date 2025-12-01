# from huggingface_hub import snapshot_download
# import os
# import shutil

# # snapshot_download(
# #     repo_id="liuhaotian/llava-bench-in-the-wild",
# #     repo_type="dataset",
# #     local_dir="./benchmarks/llavabench",
# #     local_dir_use_symlinks=False,
# #     allow_patterns=["images/**"]    # ← このパターンにマッチするファイルだけ落とす
# # )
# # snapshot_download(
# #     repo_id="Shengcao1006/MMHal-Bench",
# #     repo_type="dataset",
# #     local_dir="./datasets/mmhalbench",
# #     local_dir_use_symlinks=False,
# #     allow_patterns=["images/**"]    # ← このパターンにマッチするファイルだけ落とす
# # )
# local_dir = "./datasets/pope"
# snapshot_download(
#     repo_id="llizhx/POPE",
#     repo_type="dataset",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,
#     allow_patterns=["data/**"]    # ← このパターンにマッチするファイルだけ落とす
# )

# # # MMVP imagesフォルダの中身をlocal_dirに移動し、フォルダを削除
# # mmvp_images_dir = os.path.join(local_dir, "MMVP images")
# # if os.path.exists(mmvp_images_dir):
# #     # フォルダ内のすべてのファイルとサブディレクトリを移動
# #     for item in os.listdir(mmvp_images_dir):
# #         src = os.path.join(mmvp_images_dir, item)
# #         dst = os.path.join(local_dir, item)
# #         if os.path.exists(dst):
# #             # 既に存在する場合は上書き
# #             if os.path.isdir(dst):
# #                 shutil.rmtree(dst)
# #             else:
# #                 os.remove(dst)
# #         shutil.move(src, dst)
# #     # 空になったMMVP imagesフォルダを削除
# #     os.rmdir(mmvp_images_dir)
# #     print(f"Moved contents from 'MMVP images' to '{local_dir}' and removed the folder.")






# POPE
# from huggingface_hub import snapshot_download
# from datasets import load_dataset, Image as HfImage
# from PIL import Image
# import os

# # -----------------------------
# # 1. parquet をローカルにダウンロード
# # -----------------------------
# local_dir = "./datasets/pope"

# # snapshot_download(
# #     repo_id="llizhx/POPE",
# #     repo_type="dataset",
# #     local_dir=local_dir,
# #     local_dir_use_symlinks=False,
# #     allow_patterns=["data/**"]  # 必要部分だけ落とす
# # )

# # -----------------------------
# # 2. ローカル parquet を読み込む
# #    → split を明示して読み込む
# # -----------------------------
# data_files = {
#     "sketch_style": f"{local_dir}/data/sketch_style-00002-of-00003.parquet"
# }

# ds = load_dataset(
#     "parquet",
#     data_files=data_files,
#     split="sketch_style"
# )

# # Image 型に変換（PIL 化）
# ds = ds.cast_column("image", HfImage())

# # -----------------------------
# # 3. 元のファイル名（image_source）で保存
# # -----------------------------
# save_dir = local_dir
# os.makedirs(save_dir, exist_ok=True)

# for example in ds:
#     img: Image.Image = example["image"]

#     # 例: COCO_val2014_000000391895
#     src_name = example["image_source"]

#     # 拡張子を付ける（元データはPNG/WEBPなど混在しないので自由）
#     filename = f"{src_name}.jpg"

#     img.save(os.path.join(save_dir, filename))
