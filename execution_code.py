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
# 📘 Step 1: GPT-4 Vision APIで表画像の文字起こし（複数画像対応・コスト算出付き）
# ===============================

# --- 1️⃣ APIクライアントを初期化 ---
client = OpenAI()

# --- 2️⃣ OCR対象フォルダを指定 ---
folder_path = "cut_rows"

# --- 3️⃣ OCRプロンプト設定（Information版） ---
prompt_text_information = """
PDFもしくは画像データを文字起こしします。
文字起こしするデータは表形式で、列は以下の6列です。
カセットNo. / 薬品コード / 薬品名称 / 現在量 / 警告量 / 前回登録日

文字起こしする際、以下のルールに従ってください。
・カセットNo.は1-から始まる5桁のデータ（例: 1-001）
・薬品コードは6桁の半角数字で、00から始まります。
・薬品名称はカタカナは半角、漢字は全角で正確に記載。
・現在量と警告量は半角数字。
・前回登録日はYYYY/MM/DD形式。
・各列のデータは必ず,で区切ってください(csv形式で出力）。
・表のデータ以外（画像名や「文字起こししました」等の文章）は一切含めないでください。
・万が一、データが不明瞭で読み取れない場合は「不明」と記載してください。
"""

# --- 4️⃣ 画像ファイルの一覧を取得（Information版） ---
# folder_path にあるファイル名一覧から、Information 用の画像ファイル名だけを抽出します。
# ・os.listdir(folder_path) はフォルダ内のエントリ名（ファイル名・ディレクトリ名）を返す
# ・f.lower() で小文字化して拡張子・キーワードを大文字小文字問わず判定する
# ・endswith((".jpg", ".jpeg", ".png")) で画像拡張子のみを許可
# ・"information" in f.lower() でファイル名に "information" が含まれるものだけを選択
# ・os.path.isfile(...) でディレクトリを除外し、実際のファイルのみ対象にする
image_files_information = [
    f
    for f in os.listdir(folder_path)
    if (
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "information" in f.lower()
        and os.path.isfile(os.path.join(folder_path, f))
    )
]

# --- 5️⃣ 結果格納用のリストを作成 ---
# OCRの結果を格納するリストと、トークン使用量を合計する変数を初期化します。
results_information = []
total_prompt_tokens_information = 0
total_completion_tokens_information = 0

# --- 6️⃣ 各画像に対してOCR実行（Information版） ---
# image_files_information に含まれる各画像ファイルについてループし、
# GPT-4 Vision（chat.completions）にbase64埋め込み画像を投げてOCRを実行します。
for image_file in image_files_information:
    # 画像ファイルのフルパスを組み立て
    image_path = os.path.join(folder_path, image_file)

    # 画像ファイルをバイナリで読み込み、Base64エンコードする
    # OpenAIのVision入力では data:image/jpeg;base64,.... の形式で渡す想定
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # OpenAIクライアントにリクエストを送信
    # model は Vision 対応の軽量モデル "gpt-4.1-mini" を指定
    # messages の中で system と user を渡し、user にテキストプロンプトと画像を含める
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # Vision対応軽量モデル
        messages=[
            {
                "role": "system",
                "content": "あなたは正確なOCR変換を行う日本語文字認識エンジンです。",
            },
            {
                # user メッセージの content に配列を渡す方式（テキストと画像URLを混在）
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text_information},
                    {
                        # data URI スキームで base64 埋め込み画像を送信
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            },
        ],
        max_tokens=2000,
    )

    # --- OCR結果を取得 ---
    # API応答の先頭候補（choices[0]）からメッセージ本文を取り出す
    text_result = response.choices[0].message.content

    # --- トークン使用量を加算 ---
    # response.usage に含まれる prompt_tokens と completion_tokens を合算して後でコスト計算に使用
    usage = response.usage
    total_prompt_tokens_information += usage.prompt_tokens
    total_completion_tokens_information += usage.completion_tokens

    # --- 結果をリストに追加 ---
    # 各画像ごとにファイル名・OCRテキスト・トークン情報を辞書形式で保存
    results_information.append(
        {
            "ファイル名": image_file,
            "文字起こし結果": text_result,
            "入力トークン": usage.prompt_tokens,
            "出力トークン": usage.completion_tokens,
            "合計トークン": usage.total_tokens,
        }
    )

# --- 7️⃣ DataFrame化して表で出力 ---
# OCR結果リストを pandas DataFrame に変換して、内容を表示
df_results_information = pd.DataFrame(results_information)
print("📋 OCR結果（Information、各画像ごと）:")
print(df_results_information.to_string(index=False))

# --- 8️⃣ トークン総計とコスト計算 ---
# 合計トークン数を表示（後で任意のレートでコスト算出可能）
total_tokens_information = (
    total_prompt_tokens_information + total_completion_tokens_information
)

print("\n📊 トークン使用量（Information、合計）:")
print(f"入力トークン（prompt）: {total_prompt_tokens_information}")
print(f"出力トークン（completion）: {total_completion_tokens_information}")
print(f"合計トークン: {total_tokens_information}")

# --- 結果出力先フォルダを作成 ---
# export/result フォルダに保存するため、ディレクトリを作成（存在してもエラーにならない）
result_dir = Path("export") / "result"
result_dir.mkdir(parents=True, exist_ok=True)

# --- CSV出力パスを指定 ---
output_csv_information = result_dir / "ocr_results_information.csv"

# --- CSV保存 ---
df_results_information.to_csv(output_csv_information, index=False, encoding="utf-8-sig")

print(f"\n📁 OCR結果をCSVに保存しました: {output_csv_information.resolve()}")

# ===============================
# 📘 GPT-4 Vision APIで表画像の文字起こし（Usage版）
# ===============================

# --- 3️⃣ OCRプロンプト設定（Usage版） ---
prompt_text_usage = """
PDFもしくは画像データを文字起こしします。
文字起こしするデータは表形式で、列は以下の4列です。
薬品コード / 剤 / 薬品名称 / 使用量

文字起こしする際、以下のルールに従ってください。
・薬品コードは6桁の半角数字で、00から始まります。
・薬品名称はカタカナは半角、漢字は全角で正確に記載。
・使用量は半角数字。
・各列のデータは必ず,で区切ってください(csv形式で出力）。
・表のデータ以外（画像名や「文字起こししました」等の文章）は一切含めないでください。
・万が一読み取れない場合は「不明」と記載してください。
"""

# --- 4️⃣ 画像ファイルの一覧を取得（Usage版） ---
image_files_usage = [
    f
    for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and "usage" in f.lower()
]

# --- 5️⃣ 結果格納用のリストを作成 ---
results_usage = []
total_prompt_tokens_usage = 0
total_completion_tokens_usage = 0

# --- 6️⃣ 各画像に対してOCR実行（Usage版） ---
for image_file in image_files_usage:
    image_path = os.path.join(folder_path, image_file)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # ✅ Vision対応軽量モデル
        messages=[
            {
                "role": "system",
                "content": "あなたは正確なOCR変換を行う日本語文字認識エンジンです。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text_usage},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            },
        ],
        max_tokens=2000,
    )

    # --- OCR結果を取得 ---
    text_result = response.choices[0].message.content

    # --- トークン使用量を加算 ---
    usage = response.usage
    total_prompt_tokens_usage += usage.prompt_tokens
    total_completion_tokens_usage += usage.completion_tokens

    # --- 結果をリストに追加 ---
    results_usage.append(
        {
            "ファイル名": image_file,
            "文字起こし結果": text_result,
            "入力トークン": usage.prompt_tokens,
            "出力トークン": usage.completion_tokens,
            "合計トークン": usage.total_tokens,
        }
    )

# --- 7️⃣ DataFrame化して表で出力 ---
df_results_usage = pd.DataFrame(results_usage)
print("📋 OCR結果（Usage、各画像ごと）:")
print(df_results_usage.to_string(index=False))

# --- 8️⃣ トークン総計とコスト計算 ---
total_tokens_usage = total_prompt_tokens_usage + total_completion_tokens_usage

print("\n📊 トークン使用量（Usage、合計）:")
print(f"入力トークン（prompt）: {total_prompt_tokens_usage}")
print(f"出力トークン（completion）: {total_completion_tokens_usage}")
print(f"合計トークン: {total_tokens_usage}")


# --- CSV出力パスを指定 ---
output_csv_usage = result_dir / "Usage.csv"

# --- CSV保存 ---
df_results_usage.to_csv(output_csv_usage, index=False, encoding="utf-8-sig")

print(f"\n📁 OCR結果をCSVに保存しました: {output_csv_usage.resolve()}")

# ============================================
# OCR文字起こしの整形 → 表（6列 or 4列）に変換＋自動補正＋半角カタカナ→全角変換
# ============================================


def process_ocr_csv(csv_path, label):
    # --- 1) CSVを読み込み ---
    df_raw = pd.read_csv(csv_path)

    # --- 2) ダブルクォート削除 ---
    text_series = df_raw["文字起こし結果"].astype(str).str.strip('"')

    # --- 3) 改行で展開 ---
    tmp = pd.DataFrame(
        {"ファイル名": df_raw["ファイル名"], "文字起こし結果": text_series}
    )
    tmp["行リスト"] = (
        tmp["文字起こし結果"].str.replace("\r\n", "\n").str.replace("\r", "\n")
    )
    tmp["行リスト"] = tmp["行リスト"].str.split("\n")

    df_lines = (
        tmp[["ファイル名", "行リスト"]]
        .explode("行リスト")
        .rename(columns={"行リスト": "行テキスト"})
    )
    df_lines = df_lines[
        df_lines["行テキスト"].notna() & (df_lines["行テキスト"].str.strip() != "")
    ].copy()

    # --- 4) 補正関数（InformationとUsageで分岐）---
    def fix_line_format(line: str, label: str) -> str:
        original = line
        parts = [p.strip() for p in line.split(",")]

        # --- Information 用（カセットNoあり）---
        if label.lower() == "information":
            # カセットNo
            cassette_src = parts[0] if len(parts) >= 1 else ""
            m = re.search(r"1[-_]?(\d+)", cassette_src) or re.search(
                r"(\d+)", cassette_src
            )
            cassette_val = None
            if m:
                try:
                    cassette_val = int(m.group(1))
                except ValueError:
                    cassette_val = None
            if cassette_val is not None:
                if cassette_val > 999:
                    cassette_val = cassette_val % 1000
                parts0_fixed = f"1-{cassette_val:03d}"
            else:
                parts0_fixed = "1-000"

            # 薬品コード
            drug_src = parts[1] if len(parts) >= 2 else ""
            m2 = re.search(r"(\d+)", drug_src)
            drug_val = m2.group(1) if m2 else "000000"
            parts1_fixed = f"{int(drug_val):06d}"

            if len(parts) < 2:
                parts = [parts0_fixed, parts1_fixed]
            else:
                parts[0] = parts0_fixed
                parts[1] = parts1_fixed

        # --- Usage 用（カセットNoなし、薬品コードのみ補正）---
        elif label.lower() == "usage":
            drug_src = parts[0] if len(parts) >= 1 else ""
            m2 = re.search(r"(\d+)", drug_src)
            drug_val = m2.group(1) if m2 else "000000"
            parts0_fixed = f"{int(drug_val):06d}"
            parts[0] = parts0_fixed

        fixed = ",".join(parts)
        if fixed != original:
            print(f"🛠 修正: {original}  →  {fixed}")
        return fixed

    # --- 5) 全行に補正を適用 ---
    df_lines["行テキスト_補正後"] = df_lines["行テキスト"].apply(
        lambda x: fix_line_format(x, label)
    )

    # --- 6) 半角カタカナ → 全角カタカナ変換 ---
    df_lines["行テキスト_補正後"] = df_lines["行テキスト_補正後"].apply(
        lambda x: unicodedata.normalize("NFKC", x)
    )

    # --- 7) カンマで分割 ---
    split_df = df_lines["行テキスト_補正後"].str.split(",", expand=True)

    # --- 情報源に応じて列名を切り替え ---
    if label.lower() == "information":
        expected_cols = [
            "カセットNo",
            "薬品コード",
            "薬品名称",
            "現在量",
            "警告量",
            "前回登録日",
        ]
    elif label.lower() == "usage":
        expected_cols = ["薬品コード", "剤", "薬品名称", "使用量"]
    else:
        raise ValueError(f"Unknown label: {label}")

    # 列数が足りない場合は空列を補う
    if split_df.shape[1] < len(expected_cols):
        for _ in range(len(expected_cols) - split_df.shape[1]):
            split_df[f"欠損{_}"] = ""

    split_df = split_df.iloc[:, : len(expected_cols)]
    split_df.columns = expected_cols

    # --- 8) 前後空白除去 ---
    for c in split_df.columns:
        split_df[c] = split_df[c].astype(str).str.strip()

    # --- 9) ファイル名を付与 ---
    split_df["ファイル名"] = df_lines["ファイル名"].values

    # --- labelに応じて列順を切り替え ---
    if label.lower() == "information":
        split_df = split_df[
            [
                "ファイル名",
                "カセットNo",
                "薬品コード",
                "薬品名称",
                "現在量",
                "警告量",
                "前回登録日",
            ]
        ]
    elif label.lower() == "usage":
        split_df = split_df[["ファイル名", "薬品コード", "剤", "薬品名称", "使用量"]]

    # --- 10) バリデーション ---
    if label.lower() == "information":
        split_df["カセットNo_OK"] = (
            split_df["カセットNo"].str.fullmatch(r"1-\d{3}").notna()
        )
        split_df["薬品コード_OK"] = (
            split_df["薬品コード"].str.fullmatch(r"00\d{4}").notna()
        )
        split_df["現在量_OK"] = split_df["現在量"].str.fullmatch(r"\d+").notna()
        split_df["警告量_OK"] = split_df["警告量"].str.fullmatch(r"\d+").notna()
        split_df["前回登録日_OK"] = (
            split_df["前回登録日"].str.fullmatch(r"\d{4}/\d{2}/\d{2}|//").notna()
        )
    elif label.lower() == "usage":
        split_df["薬品コード_OK"] = (
            split_df["薬品コード"].str.fullmatch(r"00\d{4}").notna()
        )
        split_df["使用量_OK"] = (
            split_df["使用量"].astype(str).str.fullmatch(r"\d+").notna()
        )

    # --- 11) 型変換 ---
    if label.lower() == "information":
        split_df["現在量"] = pd.to_numeric(split_df["現在量"], errors="coerce").astype(
            "Int64"
        )
        split_df["警告量"] = pd.to_numeric(split_df["警告量"], errors="coerce").astype(
            "Int64"
        )
        split_df["前回登録日_dt"] = pd.to_datetime(
            split_df["前回登録日"].replace({"//": pd.NA}),
            format="%Y/%m/%d",
            errors="coerce",
        )
    elif label.lower() == "usage":
        split_df["使用量"] = (
            pd.to_numeric(split_df["使用量"], errors="coerce")
            .apply(lambda x: int(x) if pd.notna(x) else pd.NA)
            .astype("Int64")
        )

    # --- 12) 出力（整形確認） ---
    print("📋 整形後の表データ（先頭5行）:")
    print(split_df.head().to_string(index=False))

    # --- 13) 保存 ---
    output_dir = Path("export") / today_str
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{label}.csv"
    split_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n📁 保存完了: {out_csv.resolve()}")


# --- 関数呼び出し部分 ---
process_ocr_csv(r"export/result/ocr_results_information.csv", "Information")
process_ocr_csv(r"export/result/Usage.csv", "Usage")

# ============================================
# 💊 Information.csv と Usage.csv のマージ処理
#   - カセット番号ありの薬品のみ対象
#   - 使用量を月平均換算
#   - 差分・補充数を計算
#   - HTMLレポートとして出力
# ============================================

# --- 基本設定 ---
month_count = 2  # 使用量を2か月分→1か月平均に換算
output_dir = Path("export") / today_str
html_path = output_dir / "マージレポート.html"

# --- 1️⃣ CSV読み込み ---
info_df = pd.read_csv(output_dir / "Information.csv", dtype=str)
usage_df = pd.read_csv(output_dir / "Usage.csv", dtype=str)

# --- 2️⃣ 必要な列のみ抽出 ---
info_df = info_df[["カセットNo", "薬品コード", "薬品名称", "現在量", "前回登録日"]]
usage_df = usage_df[["薬品コード", "剤", "薬品名称", "使用量"]]

# --- 3️⃣ 数値列をfloat変換 ---
info_df["現在量"] = pd.to_numeric(info_df["現在量"], errors="coerce")
usage_df["使用量"] = pd.to_numeric(usage_df["使用量"], errors="coerce")

# --- 4️⃣ カセット番号ありのみ抽出 ---
info_df = info_df[
    info_df["カセットNo"].notna() & (info_df["カセットNo"].str.strip() != "")
]

# --- 5️⃣ UsageとInformationを薬品コードでマージ ---
merged = pd.merge(
    info_df, usage_df, on="薬品コード", how="inner", suffixes=("_info", "_usage")
)


# --- 6️⃣ 差分と補充数を計算 ---
def calc_replenish(row):
    usage_per_month = (
        math.floor(row["使用量"] / month_count) if pd.notna(row["使用量"]) else 0
    )
    diff = usage_per_month - (row["現在量"] if pd.notna(row["現在量"]) else 0)

    # 初期設定
    replenish = 0
    note = ""

    if diff < 30:
        replenish = 0
    elif diff < 60:
        replenish = 50
    elif diff < 110:
        replenish = 100
    elif diff < 160:
        replenish = 150
    elif diff < 210:
        replenish = 200
    elif diff < 310:
        replenish = 300
    else:
        step = math.floor((diff - 310) / 100) + 1
        replenish = 300 + (step * 100)

    # 個別ルール
    upperLimit100 = ["002402"]
    multiple21 = ["001227"]
    multiple14 = ["003035", "003084"]
    skipCalc = [
        "000113",
        "001649",
        "002913",
        "001310",
        "002510",
        "001605",
        "002747",
        "002578",
        "000004",
        "000047",
        "002374",
        "000985",
        "001146",
        "002330",
        "001367",
        "001644",
        "001751",
        "002263",
        "000477",
        "002187",
        "003622",
        "003085",
        "003201",
        "002291",
    ]

    if row["薬品コード"] in upperLimit100:
        if replenish > 100:
            replenish = 100
        note = "上限100錠"
    elif row["薬品コード"] in multiple21:
        unit = 21
        replenish = math.ceil(diff / unit) * unit
        note = "21錠シート"
    elif row["薬品コード"] in multiple14:
        unit = 14
        replenish = math.ceil(diff / unit) * unit
        note = "14錠シート"
    elif row["薬品コード"] in skipCalc:
        replenish = "B"
        note = "バラ錠あり"

    return pd.Series(
        [usage_per_month, diff, replenish, note],
        index=["使用量（月平均）", "差分", "補充数", "備考"],
    )


calc_df = merged.apply(calc_replenish, axis=1)
merged = pd.concat([merged, calc_df], axis=1)

# 「1-XXX」から数字部分（XXX）を取り出し、整数化してソートキーにする
merged["カセット番号_sort"] = (
    merged["カセットNo"].astype(str).str.replace("1-", "", regex=False).astype(int)
)
# 数値昇順でソート
merged = merged.sort_values(by="カセット番号_sort", ascending=True).reset_index(
    drop=True
)

# --- 7️⃣ HTML生成（完全改良版）---
style = """
<style>
body {
  font-family: 'Meiryo', sans-serif;
  font-size: 7pt;
  margin: 12mm;
}

/* ===== テーブル全体 ===== */
table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  border: 1px solid #999;
}

th, td {
  border: 1px solid #999;
  padding: 2px 3px;
  line-height: 1.05;
  word-wrap: break-word;
}

/* ===== 行の高さ調整 ===== */
td {
  min-height: 2.4em;  /* 2行分くらいの高さを確保 */
  height: 2.4em;      /* 固定高さとしても適用（印刷で安定） */
}

/* ===== ヘッダー ===== */
th {
  background: #f3f3f3;
  font-weight: bold;
  text-align: center;
}

/* ===== 各列の幅指定 ===== */
/* ※ 合計100%を維持するようバランス調整 */
th:nth-child(1), td:nth-child(1) { width: 70px; text-align: center; }   /* 薬品コード */
th:nth-child(2), td:nth-child(2) { width: 60px; text-align: center; }   /* カセットNo */
th:nth-child(3), td:nth-child(3) { width: 26%; text-align: left; }      /* 薬品名称（広め） */
th:nth-child(4), td:nth-child(4) { width: 55px; text-align: center; }   /* 差分 */
th:nth-child(5), td:nth-child(5) { width: 55px; text-align: center; }   /* 補充数 */
th:nth-child(6), td:nth-child(6) { width: 13%; text-align: left; }      /* 備考（やや広め） */
th:nth-child(7), td:nth-child(7) { width: 55px; text-align: center; }   /* 使用量（月平均） */
th:nth-child(8), td:nth-child(8) { width: 55px; text-align: center; }   /* 現在量 */
th:nth-child(9), td:nth-child(9) { width: 100px; text-align: center; }   /* 最終更新日 */

/* === 重要エリア（カセットNo〜備考）を淡いグレーで強調 === */
td:nth-child(2),
td:nth-child(3),
td:nth-child(4),
td:nth-child(5),
td:nth-child(6) {
  background-color: #f7f7f7 !important; /* ごく淡いグレー (#f5f5f5 〜 #f7f7f7 が自然) */
  font-weight: 500;                     /* 少しだけ太字にして視線誘導 */
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

/* 境界線をやや強調してブロック感を出す */
td:nth-child(2) {
  border-left: 2px solid #aaa;  /* 左端を濃いグレーで引き締める */
}
td:nth-child(6) {
  border-right: 2px solid #aaa; /* 備考の右側も締める */
}

/* 補充行（黄色）との重なりを自然にするため上書き設定 */
tr.highlight td:nth-child(2),
tr.highlight td:nth-child(3),
tr.highlight td:nth-child(4),
tr.highlight td:nth-child(5),
tr.highlight td:nth-child(6) {
  background-color: #fff7ce !important; /* 補充対象行は黄色を優先 */
  font-weight: bold;
}

/* ===== 行スタイル ===== */
tr.highlight td { background-color: #fff7ce !important; }  /* 黄色強調: 印刷でも残る */
tr.blue td { color: #0044cc; }
tr.noneed td { opacity: 0.6; }

/* ===== 補充数セルの特別装飾 ===== */
.replenish-highlight {
  color: red;
  font-weight: bold;
}

/* ===== 太字強調（補充行の特定列） ===== */
.bold-col {
  font-weight: bold;
}

/* ===== 印刷対応設定 ===== */
@media print {
  body { margin: 5mm; }
  th { 
    background: #f3f3f3 !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  tr.highlight td {
    background-color: #fff7ce !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  thead { display: table-header-group; }  /* ページ繰り返し */
}
</style>
"""


# --- 行ごとのクラス判定 ---
def row_class(row):
    if row["補充数"] == 0:
        return "noneed"
    elif row["補充数"] == "B":
        return "blue"
    else:
        return "highlight"


merged["row_class"] = merged.apply(row_class, axis=1)

# --- HTMLテーブルの作成（列の並びを指定） ---
html_table = """
<table>
<thead>
<tr>
  <th>薬品コード</th>
  <th>カセットNo</th>
  <th>薬品名称</th>
  <th>差分</th>
  <th>補充数</th>
  <th>備考</th>
  <th>使用量（月平均）※</th>
  <th>現在量</th>
  <th>最終更新日</th>
</tr>
</thead>
<tbody>
"""

for _, r in merged.iterrows():
    # --- 補充数セル：赤太字 ---
    if isinstance(r["補充数"], (int, float)) and r["補充数"] > 0:
        replenish_html = f"<span class='replenish-highlight'>{int(r['補充数'])}</span>"
    else:
        replenish_html = r["補充数"]

    # --- 補充がある行の場合、カセットNo・薬品名称・差分を太字に ---
    bold_start = "<span class='bold-col'>" if r["row_class"] == "highlight" else ""
    bold_end = "</span>" if r["row_class"] == "highlight" else ""

    html_table += f"""
    <tr class='{r["row_class"]}'>
        <td>{r["薬品コード"]}</td>
        <td>{bold_start}{r["カセットNo"]}{bold_end}</td>
        <td>{bold_start}{r["薬品名称_info"]}{bold_end}</td>
        <td>{bold_start}{r["差分"]}{bold_end}</td>
        <td>{replenish_html}</td>
        <td>{r["備考"]}</td>
        <td>{r["使用量（月平均）"]}</td>
        <td>{r["現在量"]}</td>
        <td>{r["前回登録日"]}</td>
    </tr>
"""
html_table += "</tbody></table>"

# --- HTML出力 ---
html_output = f"""
<html><head><meta charset='UTF-8'>{style}</head><body>
<h2>錠剤分包機　カセット補充数集計表</h2>
{html_table}
<p style='color:#555;text-align:right;font-size:8pt;margin-top:8px;'>
※ 使用量（月平均）は {month_count}ヶ月分の使用量を集計し、1ヶ月あたりに換算しています。<br>
※ 黄色行は補充対象、青文字はバラ錠扱いです。
</p>
</body></html>
"""

# --- 保存 ---
html_path.write_text(html_output, encoding="utf-8-sig")
print(f"✅ マージレポートを作成しました: {html_path.resolve()}")
