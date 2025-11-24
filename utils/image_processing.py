# =========================================
# 📘 画像処理ユーティリティモジュール
# =========================================
# 🔹目的：
# 画像の前処理（グレースケール、二値化、モルフォロジー処理）
# 表領域の検出・分割機能をモジュール化
# 他のスクリプトから import して再利用可能

import cv2
import numpy as np
from pathlib import Path


def preprocess_image(img, folder_name, output_dir):
    """
    画像の前処理を実行し、各中間ファイルを保存する関数

    パラメータ:
        img (numpy.ndarray): cv2.imread() で読み込んだ画像データ
        folder_name (str): フォルダ名（"information" または "Usage"）
        output_dir (Path): 中間ファイルの保存先ディレクトリ

    戻り値:
        dict: 各処理段階の画像データを格納
            {
                "gray": グレースケール画像,
                "binary": 二値化画像,
                "morphed": モルフォロジー処理後の画像,
                "contours": 輪郭情報,
                "img_contours": 輪郭を描画した画像
            }

    処理フロー:
        1. グレースケール変換
        2. 二値化（適応的閾値処理）
        3. モルフォロジー処理（クロージング）
        4. 輪郭抽出
        5. 各ステップの画像をJPEGで保存

    使用例:
        >>> img = cv2.imread("path/to/image.jpg")
        >>> output_dir = Path("export/process/2025-01-20_10-30-45/information_1")
        >>> result = preprocess_image(img, "information", output_dir)
        >>> gray_img = result["gray"]
    """

    # ========== Step 2: グレースケール変換 ==========
    # BGR形式（3チャンネル）の画像を白黒（1チャンネル）に変換
    # 後続の二値化処理の性能向上と計算速度の高速化に有効
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_dir / f"{folder_name}_gray.jpg"), gray)

    # ========== Step 3: 二値化 ==========
    # adaptiveThreshold：局所的な領域ごとに閾値を自動調整
    # ・cv2.ADAPTIVE_THRESH_MEAN_C：周辺の平均値を基準に閾値を設定
    # ・cv2.THRESH_BINARY_INV：反転二値化（黒背景、白文字）
    # ・blockSize=15：局所領域のサイズ（奇数推奨）
    # ・C=10：平均値から引く定数（微調整用）
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    cv2.imwrite(str(output_dir / f"{folder_name}_binary.jpg"), binary)

    # ========== Step 4: モルフォロジー処理 ==========
    # cv2.morphologyEx(MORPH_CLOSE)：小さなノイズや穴を埋める
    # ・kernel（2,2）：構造化要素のサイズ
    # ・iterations=2：処理を2回実行してノイズ除去を強化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(str(output_dir / f"{folder_name}_morphed.jpg"), morphed)

    # ========== Step 5: 輪郭抽出 ==========
    # cv2.findContours：白い領域の輪郭を検出
    # ・cv2.RETR_EXTERNAL：最外輪郭のみを取得
    # ・cv2.CHAIN_APPROX_SIMPLE：輪郭を簡潔に圧縮表現
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を描画して確認用画像を作成
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(output_dir / f"{folder_name}_contours.jpg"), img_contours)

    # 処理結果を辞書形式で返す
    return {
        "gray": gray,
        "binary": binary,
        "morphed": morphed,
        "contours": contours,
        "img_contours": img_contours,
    }


def extract_table_region(img, contours, folder_name, output_dir):
    """
    検出された輪郭から表領域を抽出し、透視変換で補正する関数

    パラメータ:
        img (numpy.ndarray): 元の画像（カラー）
        contours (list): cv2.findContours で検出された輪郭リスト
        folder_name (str): フォルダ名（"information" または "Usage"）
        output_dir (Path): 処理結果の保存先ディレクトリ

    戻り値:
        numpy.ndarray: 抽出・補正された表画像

    処理フロー:
        1. 面積が大きい輪郭を「表領域」として選別（面積 > 10000）
        2. 最大の輪郭を対象に四角形近似
        3. 透視変換で四角形を長方形に補正
        4. 補正後の画像をJPEGで保存

    使用例:
        >>> table_img = extract_table_region(img, contours, "information", output_dir)
    """

    # ========== Step 6: 表領域の抽出 ==========
    # cv2.contourArea(cnt)：輪郭の面積を計算
    # 面積が10000以上 = 表として認識する最小サイズ
    table_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]

    if not table_contours:
        # 表領域が検出されなかった場合は None を返す
        return None

    # 最も面積が大きい輪郭を表として採用
    table_cnt = max(table_contours, key=cv2.contourArea)

    # 輪郭の周囲長を計算
    peri = cv2.arcLength(table_cnt, True)

    # 輪郭を多角形で近似（cv2.approxPolyDP）
    # 0.02 * peri：近似の精度（周囲長の2%以内の誤差を許容）
    # True：輪郭は閉じた図形
    approx = cv2.approxPolyDP(table_cnt, 0.02 * peri, True)

    # ========== Step 7: 透視変換による補正 ==========
    if len(approx) == 4:
        # 四角形として認識できた場合 → 透視変換で補正
        pts = approx.reshape(4, 2)

        def order_points(pts):
            """
            4つの点を左上→右上→右下→左下の順に並べ替える補助関数
            """
            rect = np.zeros((4, 2), dtype="float32")

            # 座標の合計が最小 = 左上
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]

            # 座標の合計が最大 = 右下
            rect[2] = pts[np.argmax(s)]

            # x - y 差分が最小 = 右上
            # x - y 差分が最大 = 左下
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            return rect

        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # 表の幅と高さを計算（左辺・右辺の平均、上辺・下辺の平均）
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB))
        maxHeight = int(max(heightA, heightB))

        # 変換先の座標（正方形配置）
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # 透視変換行列を計算
        M = cv2.getPerspectiveTransform(rect, dst)

        # 透視変換を実行
        table_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    else:
        # 四角形として認識できなかった場合 → 外接矩形を使用
        x, y, w, h = cv2.boundingRect(table_cnt)
        table_img = img[y : y + h, x : x + w]

    # 処理結果を保存
    cv2.imwrite(str(output_dir / f"{folder_name}_table.jpg"), table_img)

    return table_img


def detect_and_split_rows(
    table_img, folder_name, output_dir, idx, cut_rows_dir, rows_per_split=15
):
    """
    表画像から横線を検出し、指定行数ごとに分割して保存する関数

    パラメータ:
        table_img (numpy.ndarray): 抽出された表画像
        folder_name (str): フォルダ名（"information" または "Usage"）
        output_dir (Path): 中間処理結果の保存先ディレクトリ
        idx (int): 画像の連番（1, 2, 3, ...）
        cut_rows_dir (Path): 切り抜き画像の保存先ディレクトリ
        rows_per_split (int): 何行ごとに分割するか（デフォルト: 15）

    戻り値:
        int: 切り抜き数（ファイル個数）

    処理フロー:
        1. 表画像をグレースケール化
        2. 適応的二値化で行の境界線を検出
        3. モルフォロジー処理で横線を強調
        4. 輪郭から Y 座標を抽出
        5. 指定行数ごとに画像を分割・保存

    使用例:
        >>> cut_count = detect_and_split_rows(
        ...     table_img, "information", output_dir, 1, cut_rows_dir
        ... )
        >>> print(f"切り抜きファイル数: {cut_count}")
    """

    # ========== Step 8-1: 表画像の前処理 ==========
    gray_table = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

    # 適応的二値化
    thresh_table = cv2.adaptiveThreshold(
        gray_table, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # ========== Step 8-2: 横線抽出用モルフォロジー処理 ==========
    # cv2.getStructuringElement(MORPH_RECT, (40, 1))：
    # 幅40、高さ1の細長い直線 = 横線を検出するのに最適
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

    # MORPH_OPEN：小さなノイズを除去（横線以外の不要な成分を削除）
    opened = cv2.morphologyEx(
        thresh_table, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # ========== Step 8-3: 横線の輪郭から Y 座標を抽出 ==========
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を Y 座標でソート（上から下へ）
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # 各輪郭の Y 座標を抽出
    y_positions = [cv2.boundingRect(c)[1] for c in contours]

    if len(y_positions) < 2:
        # 横線が十分に検出されなかった場合
        return 0

    # ========== Step 8-4: 指定行数ごとに分割 ==========
    cut_idx = 0

    # range(0, len(y_positions) - 1, rows_per_split)：
    # 0 から最後から1つ前まで、rows_per_split（デフォルト15）行ごとにステップ
    for cut_idx, i in enumerate(
        range(0, len(y_positions) - 1, rows_per_split), start=1
    ):
        # 分割の開始 Y 座標
        y1 = y_positions[i]

        # 分割の終了 Y 座標（15行後、ただし最後の行より後ろには行かない）
        y2 = y_positions[min(i + rows_per_split, len(y_positions) - 1)]

        # 画像をスライスして切り抜き
        row_img = table_img[y1:y2, :]

        # 保存ファイル名を作成
        # 例: information_1_1.jpg, information_1_2.jpg, ...
        save_name = f"{folder_name}_{idx}_{cut_idx}.jpg"
        save_path = cut_rows_dir / save_name

        # ファイルに保存
        cv2.imwrite(str(save_path), row_img)

    return cut_idx


def process_all_images(input_root, export_root, cut_rows_dir, folders):
    """
    全フォルダの全画像に対して前処理・表検出・分割を一括実行する関数

    パラメータ:
        input_root (Path): 入力画像フォルダのルート（例: Path("images")）
        export_root (Path): 処理結果の出力フォルダ（例: Path("export/process/2025-01-20_...")）
        cut_rows_dir (Path): 切り抜き画像の保存先フォルダ
        folders (list): 処理対象フォルダ名のリスト（例: ["information", "Usage"]）

    戻り値:
        dict: 処理結果の集計
            {
                "total_images": 処理した画像総数,
                "total_cuts": 切り抜き総数,
                "errors": エラーメッセージリスト
            }

    処理フロー:
        1. 各フォルダ内の画像ファイルをループ
        2. 各画像に対して preprocess_image() を実行
        3. 表領域を extract_table_region() で抽出
        4. 行を detect_and_split_rows() で分割
        5. エラーやスキップは統計情報として記録

    使用例:
        >>> result = process_all_images(
        ...     Path("images"),
        ...     Path("export/process/2025-01-20_10-30-45"),
        ...     Path("cut_rows"),
        ...     ["information", "Usage"]
        ... )
        >>> print(f"処理済み画像: {result['total_images']}")
        >>> print(f"作成切り抜き: {result['total_cuts']}")
    """

    total_images = 0
    total_cuts = 0
    errors = []

    for folder in folders:
        # 入力フォルダパスを作成
        input_dir = input_root / folder

        # フォルダ内の画像ファイルを取得（拡張子でフィルタリング）
        image_files = [
            f
            for f in input_dir.glob("*.*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        ]

        # 処理開始メッセージ
        print(
            f"\n=== 📂 {folder} フォルダ: {len(image_files)} 枚の画像を処理します ==="
        )

        # 画像ファイルを 1 枚ずつ処理
        for idx, img_path in enumerate(sorted(image_files), start=1):
            print(f"\n▶ 処理中: {img_path.name}")

            # 出力フォルダを作成（例: export/process/2025-01-20_10-30-45/information_1）
            output_dir = export_root / f"{folder}_{idx}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # ========== 画像の読み込み ==========
            img = cv2.imread(str(img_path))
            if img is None:
                error_msg = f"⚠ 読み込み失敗: {img_path}"
                print(error_msg)
                errors.append(error_msg)
                continue

            total_images += 1

            # ========== 前処理を実行 ==========
            preprocess_result = preprocess_image(img, folder, output_dir)

            # ========== 表領域を抽出 ==========
            table_img = extract_table_region(
                img, preprocess_result["contours"], folder, output_dir
            )

            if table_img is None:
                error_msg = f"⚠ 表領域が検出されませんでした: {img_path.name}"
                print(error_msg)
                errors.append(error_msg)
                continue

            # ========== 横線検出と行分割を実行 ==========
            cut_count = detect_and_split_rows(
                table_img, folder, output_dir, idx, cut_rows_dir
            )

            if cut_count == 0:
                error_msg = f"⚠ 横線が検出されませんでした: {img_path.name}"
                print(error_msg)
                errors.append(error_msg)
                continue

            total_cuts += cut_count

            print(f"✅ {img_path.name} → 処理完了。{cut_count}個の切り抜きを作成")

    print(f"\n🎉 全画像の処理が完了しました！")

    return {
        "total_images": total_images,
        "total_cuts": total_cuts,
        "errors": errors,
    }
