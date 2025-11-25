import pandas as pd
import re
import unicodedata
from pathlib import Path
from datetime import datetime


def clean_ocr_text(text, label):
    """
    OCR結果のテキストから、CSVとして有効な行のみを抽出する関数
    Markdown記法（```csv）や、無関係なテキストを除去します。
    """
    if not isinstance(text, str):
        return []

    # 改行コードの統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    valid_lines = []

    for line in lines:
        line = line.strip()

        # 空行やMarkdownの開始・終了タグをスキップ
        if not line:
            continue
        if line.startswith("```"):
            continue

        # カンマが含まれていない行はCSVデータではないとみなす
        if "," not in line:
            continue

        # ラベルごとの簡易チェック（明らかに形式が違う行を除外）
        parts = line.split(",")

        if label.lower() == "information":
            # Informationは通常6列。少なくとも3列以上はあるはず
            if len(parts) < 3:
                continue
        elif label.lower() == "usage":
            # Usageは通常4列。少なくとも2列以上はあるはず
            if len(parts) < 2:
                continue

        valid_lines.append(line)

    return valid_lines


def fix_line_format(line: str, label: str) -> str:
    """
    1行分のCSVテキストを補正する関数
    """
    original = line
    parts = [p.strip() for p in line.split(",")]

    # --- Information 用（カセットNoあり）---
    if label.lower() == "information":
        # カセットNo
        cassette_src = parts[0] if len(parts) >= 1 else ""
        m = re.search(r"1[-_]?(\d+)", cassette_src) or re.search(r"(\d+)", cassette_src)
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
    # if fixed != original:
    #     print(f"🛠 修正: {original}  →  {fixed}")
    return fixed


def process_ocr_csv(csv_path, label, output_root, time=None):
    """
    OCR結果のCSVを読み込み、整形して保存する関数
    """
    print(f"\n🔄 CSV整形開始: {csv_path} (Label: {label}, Time: {time})")

    # --- 1) CSVを読み込み ---
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠ CSV読み込みエラー: {e}")
        return

    # --- 2) ダブルクォート削除 ---
    # 文字起こし結果列が存在するか確認
    if "文字起こし結果" not in df_raw.columns:
        print(f"⚠ '文字起こし結果'列が見つかりません。スキップします。")
        return

    text_series = df_raw["文字起こし結果"].astype(str).str.strip('"')

    # --- 3) クリーニングと展開 ---
    # 各行のテキストをクリーニング（Markdown除去など）してからリスト化
    cleaned_lines_list = []
    filenames = []

    for idx, text in text_series.items():
        fname = df_raw.iloc[idx]["ファイル名"]
        valid_lines = clean_ocr_text(text, label)

        for line in valid_lines:
            cleaned_lines_list.append(line)
            filenames.append(fname)

    df_lines = pd.DataFrame({"ファイル名": filenames, "行テキスト": cleaned_lines_list})

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
    # print("📋 整形後の表データ（先頭5行）:")
    # print(split_df.head().to_string(index=False))

    # --- 13) 保存 ---
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名に回数を含める
    if time:
        out_csv = output_dir / f"{label}_{time}.csv"
    else:
        out_csv = output_dir / f"{label}.csv"

    split_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"📁 保存完了: {out_csv.resolve()}")
