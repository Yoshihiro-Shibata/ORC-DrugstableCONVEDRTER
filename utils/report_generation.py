import pandas as pd
import math
from pathlib import Path


def generate_merge_report(output_dir: Path, month_count: int, master_csv_path: Path):
    """
    Information.csv と Usage.csv をマージし、Master.csv との整合性をチェックした上で
    HTMLレポートを作成します。
    """
    print("\n📊 マージレポート生成プロセスを開始します...")

    html_path = output_dir / "マージレポート.html"

    # --- 1️⃣ CSV読み込み ---
    try:
        info_df = pd.read_csv(output_dir / "Information.csv", dtype=str)
        usage_df = pd.read_csv(output_dir / "Usage.csv", dtype=str)
    except FileNotFoundError as e:
        print(f"⚠ 必要なCSVファイルが見つかりません: {e}")
        return

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

    # ============================================================
    # 🛠 Master.csv との整合性チェック・対話的修正
    # ============================================================

    # 比較用の正規化関数 (数値変換できるものは数値に、それ以外は文字列のまま)
    def normalize_code(val):
        s = str(val).strip()
        try:
            return int(s)
        except ValueError:
            return s

    if master_csv_path.exists():
        print("\n🔍 Master.csv との整合性をチェックします...")
        master_df = pd.read_csv(master_csv_path, dtype=str)

        # --- 1. Master側のキー列を確実に文字列化して空白除去 ---
        # 整数として保存されている場合もあるため、比較前に正規化する
        if "カセットNo" in master_df.columns:
            master_df["カセットNo"] = master_df["カセットNo"].astype(str).str.strip()
        if "薬品コード" in master_df.columns:
            master_df["薬品コード"] = master_df["薬品コード"].astype(str).str.strip()

        # 比較用カラム作成 (Master)
        master_df["_norm_code"] = master_df["薬品コード"].apply(normalize_code)
        # 比較用カラム作成 (Merged)
        merged["_norm_code"] = merged["薬品コード"].apply(normalize_code)

        # Masterの更新フラグ
        master_updated = False

        # --- 2. Mergedデータのチェックと薬品名称の更新 ---
        # mergedの各行についてチェック
        for idx, row in merged.iterrows():
            cassette_no = str(row["カセットNo"]).strip()
            norm_code = row["_norm_code"]
            drug_name_merged = str(
                row["薬品名称_info"]
            ).strip()  # Information側の名称を基準とする

            # Masterから該当データを検索 (カセットNoと正規化コードの両方が一致)
            match_master = master_df[
                (master_df["カセットNo"] == cassette_no)
                & (master_df["_norm_code"] == norm_code)
            ]

            if not match_master.empty:
                # --- 完全一致する場合 ---
                # Masterの薬品名称で上書き (Masterを正とする)
                master_name = match_master.iloc[0]["薬品名称"]
                merged.at[idx, "薬品名称_info"] = master_name
                # print(f"   ✅ 名称更新: {drug_name_merged} -> {master_name}") # ログが多いので省略可

            else:
                # --- 一致しない場合 (カセットNo or 薬品コードの不一致) ---
                print(
                    f"\n⚠ Masterと一致しませんでした: カセットNo={cassette_no}, 薬品コード={row['薬品コード']}, 名称={drug_name_merged}"
                )

                # 部分一致の確認 (カセットNoだけ、あるいは薬品コードだけ一致するものがあるか？)
                partial_cassette = master_df[master_df["カセットNo"] == cassette_no]
                partial_code = master_df[master_df["_norm_code"] == norm_code]

                if not partial_cassette.empty:
                    print(
                        f"   💡 Master上の同じカセットNo: {partial_cassette.iloc[0].to_dict()}"
                    )
                if not partial_code.empty:
                    print(
                        f"   💡 Master上の同じ薬品コード: {partial_code.iloc[0].to_dict()}"
                    )

                choice = input("👉 Masterを書き換えますか？ (y/n): ").strip().lower()
                if choice == "y":
                    # Masterを書き換える処理
                    # 既存の部分一致があればそれを修正、なければ新規追加のフローへ

                    # もしカセットNoが一致する行があれば、その行を更新する（カセットNoは物理的にユニークなはず）
                    if not partial_cassette.empty:
                        target_idx = partial_cassette.index[0]
                        print(f"   MasterのカセットNo {cassette_no} の行を更新します。")

                        # デフォルト値を提示してユーザー入力を受け付ける
                        print(
                            "   新しい値を入力してください（エンターでデフォルト値を採用）"
                        )

                        # 薬品コード
                        default_code = str(row["薬品コード"]).strip()
                        new_code_input = input(
                            f"   薬品コード [{default_code}]: "
                        ).strip()
                        final_code = new_code_input if new_code_input else default_code

                        # 薬品名称
                        new_name_input = input(
                            f"   薬品名称 [{drug_name_merged}]: "
                        ).strip()
                        final_name = (
                            new_name_input if new_name_input else drug_name_merged
                        )

                        # Master更新
                        master_df.at[target_idx, "薬品コード"] = final_code
                        master_df.at[target_idx, "薬品名称"] = final_name
                        # 正規化カラムも更新
                        master_df.at[target_idx, "_norm_code"] = normalize_code(
                            final_code
                        )

                        master_updated = True

                        # merged側も更新したMasterの内容に合わせておく（後続処理のため）
                        merged.at[idx, "薬品名称_info"] = final_name

                    else:
                        print(
                            "   Masterに対応するカセットNoが見つかりません。新規追加扱いとします。"
                        )
                        # 新規追加ロジックへ流す
                        new_row = {
                            "カセットNo": cassette_no,
                            "薬品コード": str(row["薬品コード"]).strip(),
                            "薬品名称": drug_name_merged,
                            "警告量": "",  # デフォルト
                            "剤": "",  # デフォルト
                            "_norm_code": norm_code,
                        }
                        master_df = pd.concat(
                            [master_df, pd.DataFrame([new_row])], ignore_index=True
                        )
                        master_updated = True
                        # merged側はそのまま（drug_name_merged）でOK

                # --- 3. mergedの名称修正について ---
                # 以前はここで手動修正を聞いていたが、仕様変更により廃止。
                # Masterが存在する場合は、常にMasterの名称を優先して上書きする。
                # (上記 match_master ブロックで既に実施済み)
                # Masterにない場合や、Masterを書き換えなかった場合は、mergedの元の値が残る。

                # もしMasterを書き換えた場合、その新しいMasterの値でmergedを更新する
                # (partial_cassetteブロック内で実施済み)

        # --- 2. Masterに存在しないデータの新規追加 (mergedにあってMasterにない) ---
        # 上記ループでカバーしきれない「全くの新規」を探す
        # (カセットNoも薬品コードもMasterにないケース)

        # 効率化のため、mergedのキーセットを作成 (正規化コードを使用)
        merged_keys = set(
            zip(merged["カセットNo"].astype(str).str.strip(), merged["_norm_code"])
        )
        master_keys = set(zip(master_df["カセットNo"], master_df["_norm_code"]))

        new_keys = merged_keys - master_keys

        for c_no, n_code in new_keys:
            # 該当するmergedの行を取得
            target_row = merged[
                (merged["カセットNo"].astype(str).str.strip() == c_no)
                & (merged["_norm_code"] == n_code)
            ].iloc[0]

            d_code_raw = str(target_row["薬品コード"]).strip()
            d_name = target_row["薬品名称_info"]

            print(
                f"\n🆕 Masterに存在しないデータがあります: カセットNo={c_no}, 薬品コード={d_code_raw}, 名称={d_name}"
            )
            choice = input("👉 新しく追加しますか？ (y/n): ").strip().lower()

            if choice == "y":
                print(
                    "   各項目を入力してください（エンターでスキップして元の値を使用/空欄）"
                )

                in_c_no = input(f"   カセットNo [{c_no}]: ").strip() or c_no
                in_d_code = (
                    input(f"   薬品コード [{d_code_raw}]: ").strip() or d_code_raw
                )
                in_d_name = input(f"   薬品名称 [{d_name}]: ").strip() or d_name
                in_warn = input(f"   警告量: ").strip()
                in_zai = input(f"   剤: ").strip()

                new_data = {
                    "カセットNo": in_c_no,
                    "薬品コード": in_d_code,
                    "薬品名称": in_d_name,
                    "警告量": in_warn,
                    "剤": in_zai,
                    "_norm_code": normalize_code(in_d_code),
                }
                master_df = pd.concat(
                    [master_df, pd.DataFrame([new_data])], ignore_index=True
                )
                master_updated = True

        # --- 3. Masterにあるのにmergedにないデータの削除確認 ---
        # Masterにあってmergedにないキー
        missing_keys = master_keys - merged_keys

        if missing_keys:
            print(
                f"\n🗑 Masterにあるが今回のデータ(merged)に存在しないデータが {len(missing_keys)} 件あります。"
            )
            # 全件確認は大変なので、まとめて聞くか、1件ずつ聞くか。
            # 指示は「Masterから消去しますか？（データ）でy/n」なので、1件ずつ確認する形にする

            for c_no, n_code in missing_keys:
                # Masterの該当行を取得
                m_row = master_df[
                    (master_df["カセットNo"] == c_no)
                    & (master_df["_norm_code"] == n_code)
                ].iloc[0]
                print(
                    f"   不在データ: カセットNo={c_no}, 薬品コード={m_row['薬品コード']}, 名称={m_row['薬品名称']}"
                )

                del_choice = (
                    input("👉 Masterから消去しますか？ (y/n): ").strip().lower()
                )
                if del_choice == "y":
                    # 削除実行
                    master_df = master_df[
                        ~(
                            (master_df["カセットNo"] == c_no)
                            & (master_df["_norm_code"] == n_code)
                        )
                    ]
                    master_updated = True
                    print("   → 削除しました。")

        # --- Master保存 ---
        if master_updated:
            # 保存前に一時カラムを削除
            save_df = master_df.drop(columns=["_norm_code"], errors="ignore")
            save_df.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 Master.csv を更新しました: {master_csv_path}")
        else:
            print("\n✅ Master.csv の変更はありませんでした。")

    else:
        print(f"⚠ Master.csv が見つかりません: {master_csv_path}")

    # ============================================================
    # 📄 HTMLレポート生成
    # ============================================================

    # 「1-XXX」から数字部分（XXX）を取り出し、整数化してソートキーにする
    merged["カセット番号_sort"] = (
        merged["カセットNo"].astype(str).str.replace("1-", "", regex=False).astype(int)
    )
    # 数値昇順でソート
    merged = merged.sort_values(by="カセット番号_sort", ascending=True).reset_index(
        drop=True
    )

    # --- HTML生成（完全改良版）---
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
            replenish_html = (
                f"<span class='replenish-highlight'>{int(r['補充数'])}</span>"
            )
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
