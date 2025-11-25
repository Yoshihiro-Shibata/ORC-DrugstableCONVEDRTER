import shutil
from pathlib import Path
import os


def move_all_files_to_export(export_root: Path):
    """
    指定されたフォルダ内の画像ファイルおよびCSVファイルを
    export_root 配下へ、フォルダ構造を維持したまま移動します。

    対象:
      - images/information -> export_root/images/information
      - images/usage       -> export_root/images/usage
      - cut_rows           -> export_root/cut_rows
      - export/result      -> export_root/result

    Args:
        export_root (Path): 移動先のルートディレクトリ (例: export/process/2025-11-25_...)

    Note:
        - 移動先に同名のファイルがある場合は上書きされる可能性があります。
        - 移動元のフォルダ自体は残りますが、中身の対象ファイルはなくなります。
    """
    print("\n📦 ファイルを export フォルダへ移動します...")

    # (ソースパス, 移動先の相対パス) のリスト
    targets = [
        (Path("images") / "information", Path("images") / "information"),
        (Path("images") / "usage", Path("images") / "usage"),
        (Path("cut_rows"), Path("cut_rows")),
        (Path("export") / "result", Path("result")),
    ]

    # 対象拡張子のリスト（小文字）
    target_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",  # 画像
        ".csv",  # CSV
    }

    for src_dir, dest_rel_path in targets:
        # パスの存在確認（大文字小文字の揺らぎ対応: usage -> Usage）
        if not src_dir.exists():
            if src_dir.name == "usage":
                alt_dir = src_dir.parent / "Usage"
                if alt_dir.exists():
                    src_dir = alt_dir

            if not src_dir.exists():
                print(f"⚠ 対象フォルダが見つかりません（スキップ）: {src_dir}")
                continue

        # 移動先フォルダのパスを構築
        dest_dir = export_root / dest_rel_path

        # 移動先フォルダを作成
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ 移動先フォルダの作成に失敗しました: {dest_dir}, Error: {e}")
            continue

        # フォルダ内のファイルを走査
        files = [f for f in src_dir.iterdir() if f.is_file()]

        count = 0
        for file_path in files:
            # 拡張子チェック
            if file_path.suffix.lower() in target_extensions:
                try:
                    # 移動実行
                    shutil.move(str(file_path), str(dest_dir / file_path.name))
                    count += 1
                except Exception as e:
                    print(f"❌ ファイル移動エラー: {file_path.name} -> {e}")

        if count > 0:
            print(f"   ✅ {src_dir} から {count} 個のファイルを移動しました。")
        else:
            print(f"   ℹ {src_dir} に移動対象のファイルはありませんでした。")

    print("📦 ファイル移動処理が完了しました。")
