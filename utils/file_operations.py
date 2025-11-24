# =========================================
# 📘 ファイル操作ユーティリティモジュール
# =========================================
# 🔹目的：
# 画像ファイルの一括リネーム機能をモジュール化
# 他のスクリプトから import して再利用可能

import os
import glob


def rename_image_files(
    base_dir, folders, supported_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".gif")
):
    """
    指定フォルダ内の画像ファイルを番号付きでリネームする関数

    パラメータ:
        base_dir (str): ルートディレクトリパス（例: "images"）
        folders (list): 処理対象フォルダ名のリスト（例: ["information", "Usage"]）
        supported_extensions (tuple): 対応画像拡張子
            デフォルト: (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    戻り値:
        なし（処理結果をコンソールに表示）

    動作:
        - 各フォルダ内の画像ファイルをソート
        - 「フォルダ名_連番.拡張子」という形式にリネーム
        - 例: "information_1.jpg", "information_2.jpg", ...
        - 処理結果をコンソールに表示

    使用例:
        >>> rename_image_files("images", ["information", "Usage"])
        >>> rename_image_files("images", ["information"],
        ...                   supported_extensions=(".jpg", ".png"))
    """

    for folder in folders:
        # 🔹 ターゲットフォルダパスを組み立て
        target_path = os.path.join(base_dir, folder)

        # 🔹 指定フォルダ内の画像ファイルを取得（jpg, png, jpeg, bmp, gif対応）
        image_files = glob.glob(os.path.join(target_path, "*.*"))
        image_files = [
            f for f in image_files if f.lower().endswith(supported_extensions)
        ]

        # 🔹 1枚ずつリネーム
        # image_files リストをソート（ファイル名順）し、enumerate で番号付けして処理
        for idx, file_path in enumerate(sorted(image_files), start=1):
            # 拡張子を取得（例: ".jpg", ".png"）
            # os.path.splitext() はパスを本体と拡張子に分割し、タプル (本体, 拡張子) を返す
            # [1] で拡張子部分だけを取得
            ext = os.path.splitext(file_path)[1]

            # 新しいファイル名を作成
            # f"{folder}_{idx}{ext}" で「information_1.jpg」のような形式になる
            new_name = f"{folder}_{idx}{ext}"

            # 新しいファイルの完全なパスを作成
            # target_path（フォルダ）と new_name（ファイル名）を結合
            new_path = os.path.join(target_path, new_name)

            # 🔹 実際にファイル名を変更（旧パスから新パスへ）
            os.rename(file_path, new_path)

            # ユーザーに変更内容を通知
            print(f"✅ {file_path} → {new_path}")

    print("\n🎉 すべてのファイル名変更が完了しました！")
