#!/usr/bin/env python
"""
PokémonCryML - エンドツーエンド実行スクリプト

このスクリプトは、データ収集からモデル学習・予測までの
全パイプラインを自動的に実行します。

使い方:
    python run_full_pipeline.py [オプション]

オプション:
    --max-pokemon N    ダウンロードするポケモンの最大数（デフォルト: 100）
    --skip-download    鳴き声のダウンロードをスキップ
    --skip-training    モデル学習をスキップ
    --test-pokemon N   予測テストするポケモン名（複数指定可）
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

# カラー出力用
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message):
    """ヘッダーを表示"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}{Colors.ENDC}\n")


def print_step(step_num, message):
    """ステップを表示"""
    print(f"{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}] {message}{Colors.ENDC}")


def print_success(message):
    """成功メッセージを表示"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_error(message):
    """エラーメッセージを表示"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def print_warning(message):
    """警告メッセージを表示"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


def run_command(command, description, check=True):
    """コマンドを実行"""
    print(f"  実行中: {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            # 重要な出力のみ表示
            for line in result.stdout.split('\n'):
                if '✅' in line or 'Success' in line or 'Total' in line:
                    print(f"    {line}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description}が失敗しました")
        print(f"    エラー: {e.stderr}")
        return False


def check_dependencies():
    """依存関係をチェック"""
    print_step(0, "依存関係のチェック")
    
    # 仮想環境のチェック
    if not os.path.exists('.venv'):
        print_error("仮想環境が見つかりません")
        print("  以下のコマンドで仮想環境を作成してください:")
        print("    python -m venv .venv")
        return False
    
    print_success("仮想環境が見つかりました")
    
    # 必要なディレクトリを作成
    Path("data/cries").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    print_success("ディレクトリ構造を確認しました")
    return True


def step1_fetch_stats():
    """ステップ1: ステータスデータ取得"""
    print_step(1, "ポケモンステータスデータの取得")
    return run_command(
        "python scripts/fetch_stats.py",
        "ステータスデータのダウンロード"
    )


def step2_download_cries(max_pokemon):
    """ステップ2: 鳴き声ダウンロード"""
    print_step(2, f"ポケモン鳴き声のダウンロード (最大{max_pokemon}匹)")
    
    # download_cries.pyを一時的に修正して実行
    import importlib.util
    spec = importlib.util.spec_from_file_location("download_cries", "scripts/download_cries.py")
    module = importlib.util.module_from_spec(spec)
    
    # max_pokemonを設定して実行
    print(f"  {max_pokemon}匹の鳴き声をダウンロード中...")
    try:
        spec.loader.exec_module(module)
        module.download_pokemon_cries(max_pokemon=max_pokemon)
        print_success(f"{max_pokemon}匹の鳴き声ダウンロード完了")
        return True
    except Exception as e:
        print_error(f"鳴き声のダウンロードに失敗: {e}")
        return False


def step3_extract_features():
    """ステップ3: 音響特徴量抽出"""
    print_step(3, "音響特徴量の抽出")
    return run_command(
        "python scripts/extract_audio_features.py",
        "音響特徴量の抽出"
    )


def step4_merge_data():
    """ステップ4: データ統合"""
    print_step(4, "データの統合")
    return run_command(
        "python scripts/merge_dataset.py",
        "データの統合"
    )


def step5_train_model():
    """ステップ5: モデル学習"""
    print_step(5, "機械学習モデルの学習")
    return run_command(
        "python scripts/train_model.py",
        "モデルの学習"
    )


def step6_test_predictions(test_pokemon):
    """ステップ6: 予測テスト"""
    print_step(6, "予測のテスト")
    
    if not test_pokemon:
        # デフォルトで3匹テスト
        test_pokemon = ['pikachu', 'charmander', 'bulbasaur']
    
    success_count = 0
    for pokemon in test_pokemon:
        audio_path = f"data/cries/{pokemon}.ogg"
        if not os.path.exists(audio_path):
            print_warning(f"{pokemon}の音声ファイルが見つかりません: {audio_path}")
            continue
        
        print(f"\n  {pokemon.upper()}の予測:")
        result = subprocess.run(
            f"python scripts/predict.py {audio_path}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        # 予測結果を表示
        for line in result.stdout.split('\n'):
            if 'Predicted Stats' in line or any(stat in line.upper() for stat in ['HP', 'ATTACK', 'DEFENSE', 'SPEED']):
                print(f"    {line}")
        
        success_count += 1
    
    if success_count > 0:
        print_success(f"{success_count}匹のポケモンで予測テスト完了")
        return True
    else:
        print_error("予測テストに失敗しました")
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='PokémonCryML エンドツーエンド実行スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--max-pokemon', type=int, default=100,
                        help='ダウンロードするポケモンの最大数（デフォルト: 100）')
    parser.add_argument('--skip-download', action='store_true',
                        help='鳴き声のダウンロードをスキップ')
    parser.add_argument('--skip-training', action='store_true',
                        help='モデル学習をスキップ')
    parser.add_argument('--test-pokemon', nargs='+',
                        help='予測テストするポケモン名（例: pikachu charmander）')
    
    args = parser.parse_args()
    
    print_header("PokémonCryML - エンドツーエンド実行")
    
    # 依存関係チェック
    if not check_dependencies():
        sys.exit(1)
    
    success = True
    
    # ステップ1: ステータスデータ取得
    if success:
        success = step1_fetch_stats()
    
    # ステップ2: 鳴き声ダウンロード
    if success and not args.skip_download:
        success = step2_download_cries(args.max_pokemon)
    elif args.skip_download:
        print_warning("鳴き声のダウンロードをスキップしました")
    
    # ステップ3: 音響特徴量抽出
    if success:
        success = step3_extract_features()
    
    # ステップ4: データ統合
    if success:
        success = step4_merge_data()
    
    # ステップ5: モデル学習
    if success and not args.skip_training:
        success = step5_train_model()
    elif args.skip_training:
        print_warning("モデル学習をスキップしました")
    
    # ステップ6: 予測テスト
    if success:
        success = step6_test_predictions(args.test_pokemon)
    
    # 最終結果
    print_header("実行結果")
    if success:
        print_success("全てのステップが正常に完了しました！")
        print("\n次のステップ:")
        print("  - notebooks/analysis.ipynbでデータ分析を実行")
        print("  - python scripts/predict.py data/cries/<pokemon>.ogg で任意のポケモンを予測")
        print("  - より多くのデータで再学習: python run_full_pipeline.py --max-pokemon 500")
        return 0
    else:
        print_error("一部のステップが失敗しました")
        print("詳細はエラーメッセージを確認してください")
        return 1


if __name__ == "__main__":
    sys.exit(main())
