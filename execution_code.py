# =========================================
# 📘 画像ファイル名一括変更スクリプト
# =========================================
# 🔹目的：
# images/information と images/Usage にある画像ファイルの名前を
# 「information_◯」「Usage_◯」のように番号付きでリネームします。

import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import base64
import pandas as pd
import re
import unicodedata
import math

# 🔹 ユーティリティモジュールをインポート
from utils.file_operations import rename_image_files
from utils.image_processing import process_all_images
from utils.ocr_processing import process_ocr_by_label

# =========================================
# 📁 対象フォルダのパスを設定
# =========================================
base_dir = "images"  # ルートディレクトリ
folders = ["information", "Usage"]  # 対象サブフォルダ

# =========================================
# 📂 画像ファイルの一括リネーム実行
# =========================================
rename_image_files(base_dir, folders)

# ============================================================
# 📘 画像一括処理：表検出・分割・cut_rows保存対応版
# ============================================================

# =========================================
# 📁 入出力ディレクトリ設定
# =========================================
input_root = Path("images")

# export/process/日付_時刻 のフォルダを作成
# 現在時刻を文字列化（フォルダ名に使用）
today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# =========================================
# 📁 出力先ディレクトリの準備
# =========================================

# 🔹 出力先のルートパスを組み立て
# Path オブジェクトを使うことでOSに依存しないパス操作が可能
# 「export/process/<実行日時>」フォルダ構造を作る
export_root = Path("export") / "process" / today_str

# 🔹 フォルダを作成（親ディレクトリも含めて再帰的に作成）
# parents=True → 途中のディレクトリ（export, process）も自動作成
# exist_ok=True → 既にフォルダが存在していてもエラーにならない
export_root.mkdir(parents=True, exist_ok=True)

# 🔹 切り抜き画像（15行ごとに分割したもの）を保存するフォルダ
# Path型にすることで、後で「/」演算子で結合できる（直感的で安全）
cut_rows_dir = Path("cut_rows")

# 🔹 cut_rowsフォルダも同様に作成
cut_rows_dir.mkdir(parents=True, exist_ok=True)


# =========================================
# 📦 各フォルダ内の画像を順に処理（モジュール化版）
# =========================================
# 🔹 utils.image_processing モジュールに実装された process_all_images() を呼び出し
# 画像前処理・表検出・横線検出・分割を一括処理する
#
# 戻り値:
#   - total_images: 処理した画像の総数
#   - total_cuts: 作成した切り抜きファイルの総数
#   - errors: 処理中に発生したエラーメッセージのリスト

result = process_all_images(input_root, export_root, cut_rows_dir, folders)


# ===============================
# 📘 Step 1: OCR処理（ラベル分岐版・モジュール化・2回実行）
# ===============================
# 🔹 utils.ocr_processing モジュールを使用
# 信頼性向上のため、異なるモデルで2回OCRを実行し、結果を保存する
# 1回目: gpt-4.1-mini (軽量・高速)
# 2回目: gpt-4o (高精度・変化をつける)

result_dir = Path("export") / "result"

# OCR設定（回数とモデルの組み合わせ）
ocr_configs = [
    {"time": 1, "model": "gpt-4.1-mini"},
    {"time": 2, "model": "gpt-4.1-mini"},
]

ocr_results_list = []

for config in ocr_configs:
    print(f"\n🔄 OCR実行 {config['time']}回目 (モデル: {config['model']})...")
    ocr_result = process_ocr_by_label(
        cut_rows_dir, result_dir, model=config["model"], time=config["time"]
    )
    ocr_results_list.append(ocr_result)

# 後続処理のために、とりあえず1回目の結果を使用する（または比較ロジックをここに実装する）
# 現時点では1回目の結果（ocr_results_list[0]）を使って後続のCSV整形を行う
# ============================================
# OCR文字起こしの整形 → 表（6列 or 4列）に変換＋自動補正＋半角カタカナ→全角変換
# ============================================
from utils.data_processing import process_ocr_csv
from utils.comparison import compare_and_merge_csvs

# 保存先ディレクトリ
final_output_dir = Path("export") / today_str

# 各回のOCR結果を順に処理
for i, ocr_result in enumerate(ocr_results_list):
    time_index = i + 1
    print(f"\n🔄 {time_index}回目のOCR結果を整形中...")

    if ocr_result["information_csv"]:
        process_ocr_csv(
            ocr_result["information_csv"],
            "Information",
            final_output_dir,
            time=time_index,
        )

    if ocr_result["usage_csv"]:
        process_ocr_csv(
            ocr_result["usage_csv"], "Usage", final_output_dir, time=time_index
        )

# ============================================
# 🔍 2回のOCR結果を比較・検証・マージ
# ============================================
compare_and_merge_csvs(final_output_dir)

# ============================================
# 💊 Information.csv と Usage.csv のマージ処理
#   - カセット番号ありの薬品のみ対象
#   - 使用量を月平均換算
#   - 差分・補充数を計算
#   - Master.csv との整合性チェック・対話的修正
#   - HTMLレポートとして出力
# ============================================
from utils.report_generation import generate_merge_report

# --- 基本設定 ---
month_count = 2  # 使用量を2か月分→1か月平均に換算
master_csv_path = Path("Master.csv")

# レポート生成モジュールの呼び出し
generate_merge_report(final_output_dir, month_count, master_csv_path)


# ============================================
# 📦 画像ファイルの移動処理
# ============================================
# 処理が終わった画像ファイルを export フォルダへ移動して整理する
from utils.image_mover import move_all_files_to_export

# export_root はコード前半で定義済み (export/process/YYYY-MM-DD_HH-MM-SS)
move_all_files_to_export(export_root)
