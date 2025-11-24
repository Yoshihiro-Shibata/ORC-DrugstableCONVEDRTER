# =========================================
# 📘 フォルダラベリングユーティリティモジュール
# =========================================
# 🔹目的：
# 対象ディレクトリ内のフォルダ名を調査し、
# フォルダ名に含まれるキーワードに基づいてラベルを返す機能をモジュール化
# 他のスクリプトから import して再利用可能

from pathlib import Path


def get_folder_label(folder_name):
    """
    フォルダ名からラベルを判定する関数

    パラメータ:
        folder_name (str): フォルダの名称

    戻り値:
        str: 判定結果のラベル
            - "information": フォルダ名に "information" が含まれる場合
            - "usage": フォルダ名に "usage" が含まれる場合
            - "others": 上記以外の場合

    判定ルール:
        ・大文字小文字を区別しない（例: "Information", "INFORMATION", "information" すべて "information" と判定）
        ・部分一致で判定（例: "my_information_folder" は "information" と判定）
        ・"information" を優先チェック（両方含まれる場合は "information" を返す）
        ・"usage" 次点でチェック
        ・どちらも含まれない場合は "others" を返す

    使用例:
        >>> label = get_folder_label("information_data")
        >>> print(label)  # 出力: "information"

        >>> label = get_folder_label("usage_logs")
        >>> print(label)  # 出力: "usage"

        >>> label = get_folder_label("backup_files")
        >>> print(label)  # 出力: "others"
    """

    # フォルダ名を小文字に統一して判定
    folder_lower = folder_name.lower()

    # "information" が含まれるか判定（優先度：1）
    if "information" in folder_lower:
        return "information"

    # "usage" が含まれるか判定（優先度：2）
    if "usage" in folder_lower:
        return "usage"

    # どちらも含まれない場合
    return "others"


def label_folders_in_directory(directory_path):
    """
    指定ディレクトリ内のすべてのフォルダをスキャンし、
    各フォルダにラベルを付与した結果を辞書形式で返す関数

    パラメータ:
        directory_path (str または Path): スキャン対象のディレクトリパス
            例: "export/process/2025-01-20_10-30-45"

    戻り値:
        dict: フォルダ名とラベルのマッピング
            {
                "information_1": "information",
                "information_2": "information",
                "Usage_1": "usage",
                "backup": "others"
            }

    処理フロー:
        1. ディレクトリパスを Path オブジェクトに変換
        2. ディレクトリ内のすべての項目を列挙
        3. ディレクトリ（フォルダ）のみを対象に
        4. 各フォルダ名に get_folder_label() を適用
        5. 結果を辞書にまとめて返す

    使用例:
        >>> result = label_folders_in_directory("export/process/2025-01-20_10-30-45")
        >>> print(result)
        {
            'information_1': 'information',
            'information_2': 'information',
            'Usage_1': 'usage',
            'Usage_2': 'usage'
        }

        >>> for folder_name, label in result.items():
        ...     print(f"{folder_name}: {label}")
    """

    # ディレクトリパスを Path オブジェクトに統一
    dir_path = Path(directory_path)

    # パスが存在し、ディレクトリであることを確認
    if not dir_path.exists():
        print(f"⚠ エラー: ディレクトリが見つかりません - {dir_path}")
        return {}

    if not dir_path.is_dir():
        print(f"⚠ エラー: パスがディレクトリではありません - {dir_path}")
        return {}

    # 結果を格納する辞書
    label_dict = {}

    # ディレクトリ内のすべての項目を列挙
    for item in dir_path.iterdir():
        # ディレクトリのみを対象
        if item.is_dir():
            # フォルダ名を取得
            folder_name = item.name

            # ラベルを判定
            label = get_folder_label(folder_name)

            # 結果に追加
            label_dict[folder_name] = label

    return label_dict


def print_folder_labels(directory_path, show_details=False):
    """
    指定ディレクトリ内のフォルダをスキャンし、
    ラベル付け結果をコンソールに見やすく表示する関数

    パラメータ:
        directory_path (str または Path): スキャン対象のディレクトリパス
        show_details (bool): 詳細情報を表示するか
            True: 各フォルダ名と対応するラベルを個別表示
            False: ラベルごとの統計情報を表示（デフォルト）

    戻り値:
        dict: フォルダ名とラベルのマッピング

    処理フロー:
        1. label_folders_in_directory() を呼び出し
        2. show_details が True の場合、全フォルダの詳細を表示
        3. show_details が False の場合、ラベルごとの統計を表示
        4. 結果の辞書を返す

    使用例:
        >>> # 詳細表示
        >>> result = print_folder_labels("export/process/2025-01-20_10-30-45", show_details=True)

        >>> # 統計表示
        >>> result = print_folder_labels("export/process/2025-01-20_10-30-45", show_details=False)
    """

    # ラベル付けを実行
    label_dict = label_folders_in_directory(directory_path)

    # 結果が空の場合
    if not label_dict:
        print("📁 スキャン結果: フォルダが見つかりません")
        return label_dict

    print(f"\n📂 ディレクトリ: {directory_path}")
    print(f"📊 スキャン結果: 合計 {len(label_dict)} 個のフォルダを検出\n")

    if show_details:
        # ========== 詳細表示モード ==========
        print("📋 各フォルダのラベル:")
        print("-" * 60)

        # ラベルごとにグループ化して表示
        for label in ["information", "usage", "others"]:
            matching_folders = [
                folder for folder, lbl in label_dict.items() if lbl == label
            ]

            if matching_folders:
                # 🏷️ information / 🔧 usage / 📦 others の絵文字を使い分け
                if label == "information":
                    emoji = "📋"
                elif label == "usage":
                    emoji = "📊"
                else:
                    emoji = "📦"

                print(f"\n{emoji} [{label.upper()}] ({len(matching_folders)} 個)")
                for folder in sorted(matching_folders):
                    print(f"   └─ {folder}")

    else:
        # ========== 統計表示モード ==========
        print("📈 ラベル別の統計:")
        print("-" * 60)

        # ラベルごとの個数を集計
        label_counts = {}
        for label in label_dict.values():
            label_counts[label] = label_counts.get(label, 0) + 1

        # 統計情報を表示
        for label in ["information", "usage", "others"]:
            count = label_counts.get(label, 0)
            percentage = (count / len(label_dict) * 100) if label_dict else 0

            if label == "information":
                emoji = "📋"
            elif label == "usage":
                emoji = "📊"
            else:
                emoji = "📦"

            print(f"{emoji} {label:12s}: {count:3d} 個 ({percentage:5.1f}%)")

    print()
    return label_dict
