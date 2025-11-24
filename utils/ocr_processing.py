# =========================================
# 📘 OCR処理ユーティリティモジュール
# =========================================
# 🔹目的：
# GPT-4 Vision APIを使用した表画像の文字起こし処理をモジュール化
# ファイル名に基づいてラベルを判定し、適切なプロンプトを使い分ける
# 他のスクリプトから import して再利用可能

import os
import base64
from pathlib import Path
from openai import OpenAI
import pandas as pd
import cv2  # 🔹追加：画像サイズ確認用
from utils.folder_labeling import get_folder_label


# =========================================
# 📝 OCRプロンプトテンプレート
# =========================================

PROMPT_INFORMATION = """
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

PROMPT_USAGE = """
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


def get_prompt_for_label(label):
    """
    ラベルに対応するOCRプロンプトを返す関数

    パラメータ:
        label (str): フォルダラベル
            "information": Information表用プロンプト
            "usage": Usage表用プロンプト
            "others": デフォルトはInformation用を返す

    戻り値:
        str: 対応するOCRプロンプトテキスト

    使用例:
        >>> prompt = get_prompt_for_label("information")
        >>> print(prompt[:50])  # 最初の50文字
    """
    if label == "usage":
        return PROMPT_USAGE
    else:
        # "information" と "others" は Information 用プロンプトを返す
        return PROMPT_INFORMATION


def encode_image_to_base64(image_path):
    """
    画像ファイルをBase64エンコードする関数
    """
    # 🔹追加デバッグ：画像解像度とアスペクト比を確認
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            aspect_ratio = w / h if h > 0 else 0
            print(f"🔍 [DEBUG] 画像解像度: {w}x{h} (アスペクト比: {aspect_ratio:.2f})")

            # 警告：アスペクト比が極端な場合（横長すぎる場合）
            if aspect_ratio > 4.0:
                print(
                    f"⚠️ [WARNING] アスペクト比が非常に横長です。APIによる自動拡大でトークンが急増している可能性があります。"
                )
    except Exception as e:
        print(f"⚠️ [DEBUG] 画像解像度の取得に失敗: {e}")

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 🔹デバッグ用：画像サイズ（バイト数）を表示
    print(f"🔍 [DEBUG] 画像ファイルサイズ: {len(img_bytes)} bytes ({image_path})")

    return base64.b64encode(img_bytes).decode("utf-8")


def perform_ocr_on_image(client, image_path, label, model="gpt-4.1-mini"):
    """
    単一の画像ファイルに対してOCR処理を実行する関数

    パラメータ:
        client (OpenAI): 初期化済みのOpenAIクライアント
        image_path (str): 画像ファイルのパス
        label (str): フォルダラベル（"information", "usage", "others"）
        model (str): 使用するGPTモデル名（デフォルト: "gpt-4.1-mini"）

    戻り値:
        dict: OCR処理の結果
            {
                "image_file": ファイル名,
                "label": ラベル,
                "text_result": OCR結果テキスト,
                "prompt_tokens": 入力トークン数,
                "completion_tokens": 出力トークン数,
                "total_tokens": 合計トークン数
            }

    処理フロー:
        1. ラベルに基づいてOCRプロンプトを選択
        2. 画像をBase64エンコード
        3. OpenAI APIにリクエスト送信
        4. レスポンスからテキストとトークン情報を抽出
        5. 結果を辞書で返す

    使用例:
        >>> client = OpenAI()
        >>> result = perform_ocr_on_image(
        ...     client,
        ...     "cut_rows/information_1_1.jpg",
        ...     "information"
        ... )
        >>> print(result["text_result"][:100])
    """

    # ラベルに基づいてプロンプトを取得
    prompt = get_prompt_for_label(label)

    # 画像ファイルをBase64エンコード（リサイズなし）
    img_base64 = encode_image_to_base64(image_path)

    # 🔹デバッグ用：Base64文字列の長さを表示
    print(f"🔍 [DEBUG] Base64文字数: {len(img_base64)}")

    # APIリクエストのパラメータを構築
    api_params = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "あなたは正確なOCR変換を行う日本語文字認識エンジンです。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            },
        ],
    }

    # モデル名に応じてトークン制限のパラメータ名を切り替え
    # o1系列やgpt-5系列は max_completion_tokens を使用
    if model.startswith("o1") or "gpt-5" in model:
        api_params["max_completion_tokens"] = 5000
    else:
        # gpt-4, gpt-3.5 などは max_tokens を使用
        api_params["max_tokens"] = 2000

    # OpenAI APIにリクエストを送信
    response = client.chat.completions.create(**api_params)

    # レスポンスからテキストとトークン情報を抽出
    text_result = response.choices[0].message.content
    usage = response.usage

    return {
        "ファイル名": os.path.basename(image_path),
        "label": label,
        "文字起こし結果": text_result,
        "入力トークン": usage.prompt_tokens,
        "出力トークン": usage.completion_tokens,
        "合計トークン": usage.total_tokens,
    }


def process_ocr_by_label(cut_rows_dir, output_dir, model="gpt-4.1-mini", time=1):
    """
    cut_rows_dir 内のすべての画像ファイルをスキャンし、
    ラベルに基づいて分岐処理でOCR実行・結果を保存する関数

    パラメータ:
        cut_rows_dir (str または Path): 切り抜き画像が格納されているディレクトリ
        output_dir (str または Path): OCR結果を保存するディレクトリ
        model (str): 使用するGPTモデル名（デフォルト: "gpt-4.1-mini"）
        time (int): 実行回数（ファイル名に付与される）

    戻り値:
        dict: 処理結果の統計情報
            {
                "total_files": 処理したファイル総数,
                "information_count": Information系ファイル数,
                "usage_count": Usage系ファイル数,
                "others_count": その他ファイル数,
                "information_results": Informationの処理結果リスト,
                "usage_results": Usageの処理結果リスト,
                "information_csv": Information CSV保存パス,
                "usage_csv": Usage CSV保存パス
            }

    処理フロー:
        1. cut_rows_dir内の画像ファイルをスキャン
        2. 各ファイル名からラベルを判定（folder_labeling モジュール使用）
        3. ラベルに基づいて適切なプロンプトでOCR実行
        4. Information と Usage に分類して結果を格納
        5. 各ラベル別にCSV保存
        6. 統計情報を返す

    使用例:
        >>> result = process_ocr_by_label(
        ...     Path("cut_rows"),
        ...     Path("export") / "result"
        ... )
        >>> print(f"Information: {result['information_count']} 件")
        >>> print(f"Usage: {result['usage_count']} 件")
    """

    # OpenAI クライアントを初期化
    client = OpenAI()

    # パスをPath型に統一
    cut_rows_path = Path(cut_rows_dir)
    output_path = Path(output_dir)

    # 出力ディレクトリを作成
    output_path.mkdir(parents=True, exist_ok=True)

    # 結果格納用のリスト
    information_results = []
    usage_results = []
    others_results = []

    # cut_rows_dir内の画像ファイルをスキャン
    # 対応拡張子: jpg, jpeg, png
    image_files = (
        list(cut_rows_path.glob("*.[jJ][pP][gG]"))
        + list(cut_rows_path.glob("*.[jJ][pP][eE][gG]"))
        + list(cut_rows_path.glob("*.[pP][nN][gG]"))
    )

    if not image_files:
        print(f"⚠ 警告: {cut_rows_path} に画像ファイルが見つかりません")
        return {
            "total_files": 0,
            "information_count": 0,
            "usage_count": 0,
            "others_count": 0,
            "information_results": [],
            "usage_results": [],
            "information_csv": None,
            "usage_csv": None,
        }

    print(f"\n🔍 スキャン開始: {len(image_files)} 個の画像ファイルを検出\n")

    # 各画像ファイルに対してOCR処理を実行
    for idx, image_path in enumerate(sorted(image_files), start=1):
        image_file = image_path.name

        # ファイル名からラベルを判定
        label = get_folder_label(image_file)

        # 🔹【修正】others（判定不能ファイル）はOCRせずにスキップする
        if label == "others":
            print(f"📦 [{idx}/{len(image_files)}] スキップ: {image_file} (対象外)")
            others_results.append(
                {
                    "ファイル名": image_file,
                    "label": "others",
                    "文字起こし結果": "",
                    "入力トークン": 0,
                    "出力トークン": 0,
                    "合計トークン": 0,
                }
            )
            continue

        # 処理状況を表示
        emoji = "📋" if label == "information" else "📊"
        print(
            f"{emoji} [{idx}/{len(image_files)}] 処理中: {image_file} (ラベル: {label})"
        )

        try:
            # OCR処理を実行
            result = perform_ocr_on_image(client, str(image_path), label, model)

            # ラベルに基づいて結果を分類
            if label == "information":
                information_results.append(result)
            elif label == "usage":
                usage_results.append(result)
            else:
                others_results.append(result)

        except Exception as e:
            print(f"   ⚠ エラー: {str(e)}")
            continue

    # ========== Information 結果の処理 ==========
    information_csv_path = None
    if information_results:
        df_information = pd.DataFrame(information_results)

        print(f"\n📋 OCR結果（Information、全{len(information_results)}件）:")
        print(
            df_information[
                ["ファイル名", "入力トークン", "出力トークン", "合計トークン"]
            ].to_string(index=False)
        )

        # トークン統計
        total_prompt_info = df_information["入力トークン"].sum()
        total_completion_info = df_information["出力トークン"].sum()
        total_tokens_info = df_information["合計トークン"].sum()

        print(f"\n📊 トークン使用量（Information、合計）:")
        print(f"   入力トークン: {total_prompt_info}")
        print(f"   出力トークン: {total_completion_info}")
        print(f"   合計トークン: {total_tokens_info}")

        # CSV保存
        information_csv_path = output_path / f"ocr_results_information_{time}.csv"
        df_information.to_csv(information_csv_path, index=False, encoding="utf-8-sig")
        print(f"   📁 保存: {information_csv_path.resolve()}")

    # ========== Usage 結果の処理 ==========
    usage_csv_path = None
    if usage_results:
        df_usage = pd.DataFrame(usage_results)

        print(f"\n📋 OCR結果（Usage、全{len(usage_results)}件）:")
        print(
            df_usage[
                ["ファイル名", "入力トークン", "出力トークン", "合計トークン"]
            ].to_string(index=False)
        )

        # トークン統計
        total_prompt_usage = df_usage["入力トークン"].sum()
        total_completion_usage = df_usage["出力トークン"].sum()
        total_tokens_usage = df_usage["合計トークン"].sum()

        print(f"\n📊 トークン使用量（Usage、合計）:")
        print(f"   入力トークン: {total_prompt_usage}")
        print(f"   出力トークン: {total_completion_usage}")
        print(f"   合計トークン: {total_tokens_usage}")

        # CSV保存
        usage_csv_path = output_path / f"ocr_results_usage_{time}.csv"
        df_usage.to_csv(usage_csv_path, index=False, encoding="utf-8-sig")
        print(f"   📁 保存: {usage_csv_path.resolve()}")

    # ========== Others 結果の処理 ==========
    if others_results:
        print(f"\n📦 その他のファイル（{len(others_results)}件）: 処理対象外")

    # ========== 統計情報の表示 ==========
    print(f"\n📈 処理統計:")
    print(f"   Information: {len(information_results)} 件")
    print(f"   Usage: {len(usage_results)} 件")
    print(f"   Others: {len(others_results)} 件")
    print(f"   合計: {len(image_files)} 件")

    return {
        "total_files": len(image_files),
        "information_count": len(information_results),
        "usage_count": len(usage_results),
        "others_count": len(others_results),
        "information_results": information_results,
        "usage_results": usage_results,
        "information_csv": information_csv_path,
        "usage_csv": usage_csv_path,
    }
