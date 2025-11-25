import pandas as pd
from pathlib import Path


def compare_and_merge_csvs(output_dir: Path):
    """
    指定されたディレクトリ内の _1.csv と _2.csv を比較し、
    ユーザー対話形式で整合性をチェックした上で、最終的な {label}.csv を生成します。
    """
    print("\n🔍 OCR結果の比較・検証を開始します...")

    # 対象ラベル
    labels = ["Information", "Usage"]

    for label in labels:
        file1 = output_dir / f"{label}_1.csv"
        file2 = output_dir / f"{label}_2.csv"

        # 両方のファイルが存在する場合のみ比較を行う
        if not file1.exists() or not file2.exists():
            print(f"⚠ {label} の比較対象ファイルが揃っていません。スキップします。")
            continue

        print(f"\n📊 {label} データの比較中...")

        # CSV読み込み (全て文字列として読み込む)
        df1 = pd.read_csv(file1, dtype=str)
        df2 = pd.read_csv(file2, dtype=str)

        # マージキーと検証対象カラムの設定
        if label.lower() == "information":
            merge_key = "カセットNo"
            # 検証対象: 薬品コード, 現在量, 前回登録日 (薬品名称、警告量は指示に含まれていないため除外)
            # ※指示: "薬品コード、現在量、前回登録日において値が一致していることを確認"
            check_cols = ["薬品コード", "現在量", "前回登録日"]

        elif label.lower() == "usage":
            merge_key = "薬品コード"
            # 検証対象: 使用量
            check_cols = ["使用量"]
        else:
            continue

        # マージ実行 (outer結合で両方のデータを保持)
        # suffixes=('_1', '_2') で列名を区別
        merged = pd.merge(
            df1, df2, on=merge_key, how="outer", suffixes=("_1", "_2"), indicator=True
        )

        final_rows = []

        # 1行ずつチェック
        for idx, row in merged.iterrows():
            key_val = row[merge_key]

            # ケース1: 両方にデータが存在する場合 (both)
            if row["_merge"] == "both":
                is_match = True
                diff_cols = []

                for col in check_cols:
                    val1 = str(row[f"{col}_1"]).strip()
                    val2 = str(row[f"{col}_2"]).strip()

                    # 値が不一致の場合
                    if val1 != val2:
                        is_match = False
                        diff_cols.append(col)

                if is_match:
                    # 全て一致 -> index 1 のデータを採用
                    # _1 の列名を持つデータを元の列名に戻して追加
                    row_data = {c: row[f"{c}_1"] for c in df1.columns if c != merge_key}
                    row_data[merge_key] = key_val
                    # ファイル名なども _1 のものを採用
                    final_rows.append(row_data)
                else:
                    # 不一致あり -> ユーザー選択
                    print(f"\n⚡ 不一致検出 ({label}) キー: {key_val}")
                    print(f"   不一致項目: {', '.join(diff_cols)}")

                    # 表示したい追加項目（薬品名称、現在量、前回登録日など）
                    # check_cols に含まれていないものも表示対象にする
                    extra_display_cols = ["薬品名称", "現在量", "前回登録日"]
                    # 重複を除去しつつ、diff_cols にないものを追加
                    display_target_cols = list(set(diff_cols + extra_display_cols))

                    # 1回目と2回目の値を整形して表示
                    def format_vals(suffix):
                        vals = []
                        for c in display_target_cols:
                            # カラムが存在するかチェック
                            col_key = f"{c}{suffix}"
                            if col_key in row:
                                val = row[col_key]
                                # 不一致項目は強調（★）をつけるなどの工夫も可能だが、今回はシンプルに表示
                                mark = "★" if c in diff_cols else ""
                                vals.append(f"{mark}{c}={val}")
                        return ", ".join(vals)

                    print(f"   [1] 1回目: {format_vals('_1')}")
                    print(f"   [2] 2回目: {format_vals('_2')}")

                    while True:
                        choice = input("👉 どちらを採用しますか？ (1/2): ").strip()
                        if choice == "1":
                            row_data = {
                                c: row[f"{c}_1"] for c in df1.columns if c != merge_key
                            }
                            row_data[merge_key] = key_val
                            final_rows.append(row_data)
                            break
                        elif choice == "2":
                            row_data = {
                                c: row[f"{c}_2"] for c in df1.columns if c != merge_key
                            }
                            row_data[merge_key] = key_val
                            final_rows.append(row_data)
                            break
                        else:
                            print("1 か 2 を入力してください。")

            # ケース2: 片方のみ存在する場合 (left_only / right_only)
            else:
                side = "1回目のみ" if row["_merge"] == "left_only" else "2回目のみ"
                suffix = "_1" if row["_merge"] == "left_only" else "_2"
                other_suffix = "_2" if row["_merge"] == "left_only" else "_1"

                print(f"\n❓ 片方のみ検出 ({label}) キー: {key_val} ({side})")

                # 内容を表示
                display_cols = (
                    check_cols + ["薬品名称"]
                    if "薬品名称" not in check_cols
                    else check_cols
                )
                display_vals = []
                for c in display_cols:
                    col_name = f"{c}{suffix}" if f"{c}{suffix}" in row else c
                    if col_name in row:
                        display_vals.append(f"{c}={row[col_name]}")

                print(f"   内容: {', '.join(display_vals)}")

                # Informationの場合、薬品コードで反対側のデータを検索
                if label.lower() == "information":
                    drug_code_col = f"薬品コード{suffix}"
                    if drug_code_col in row:
                        target_drug_code = str(row[drug_code_col]).strip()

                        # 反対側のデータフレーム (df2 if left_only else df1)
                        other_df = df2 if row["_merge"] == "left_only" else df1

                        # 薬品コードで検索
                        # other_df は元の列名を持っているので "薬品コード" で検索
                        if "薬品コード" in other_df.columns:
                            match_rows = other_df[
                                other_df["薬品コード"].astype(str).str.strip()
                                == target_drug_code
                            ]

                            if not match_rows.empty:
                                print(
                                    f"   💡 ヒント: 反対側のデータに同じ薬品コード ({target_drug_code}) が見つかりました:"
                                )
                                for _, m_row in match_rows.iterrows():
                                    # 表示項目: カセットNo, 薬品コード, 現在量, 前回登録日
                                    hint_cols = [
                                        "カセットNo",
                                        "薬品コード",
                                        "現在量",
                                        "前回登録日",
                                    ]
                                    hint_vals = [
                                        f"{c}={m_row.get(c, 'N/A')}" for c in hint_cols
                                    ]
                                    print(f"      → {', '.join(hint_vals)}")
                            else:
                                print(
                                    f"   ⚠ 反対側のデータに同じ薬品コード ({target_drug_code}) は見つかりませんでした。"
                                )

                while True:
                    choice = (
                        input("👉 このデータを採用しますか？ (y/n): ").strip().lower()
                    )
                    if choice == "y":
                        # 採用する場合、その側のデータを使う
                        # カラム名から suffix を除去して辞書化
                        base_df = df1 if row["_merge"] == "left_only" else df2
                        row_data = {}
                        for col in base_df.columns:
                            if col == merge_key:
                                row_data[col] = key_val
                            else:
                                # マージ後の列名には suffix がついている
                                val = row.get(f"{col}{suffix}")
                                # マージキー以外で共通列でない場合（片方にしかない列など）はそのまま取れるかも？
                                # outer join なので共通列は必ず suffix がつく
                                row_data[col] = val

                        final_rows.append(row_data)
                        break
                    elif choice == "n":
                        print("   → 除外しました。")
                        break
                    else:
                        print("y か n を入力してください。")

        # 最終的なDataFrameを作成
        if final_rows:
            final_df = pd.DataFrame(final_rows)
            # 列順を元のdf1に合わせる
            final_df = final_df[df1.columns]

            # 保存
            save_path = output_dir / f"{label}.csv"
            final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ {label} の検証完了。保存しました: {save_path}")
        else:
            print(f"\n⚠ {label} の有効なデータがありませんでした。")
