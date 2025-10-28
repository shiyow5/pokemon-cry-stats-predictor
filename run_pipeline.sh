#!/bin/bash
# PokémonCryML - エンドツーエンド実行スクリプト（シェル版）

set -e  # エラーが発生したら停止

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ヘッダー表示
print_header() {
    echo -e "\n${CYAN}${BOLD}============================================================"
    echo -e "  $1"
    echo -e "============================================================${NC}\n"
}

# ステップ表示
print_step() {
    echo -e "${BLUE}${BOLD}[Step $1] $2${NC}"
}

# 成功メッセージ
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# エラーメッセージ
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 警告メッセージ
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# メイン処理
main() {
    print_header "PokémonCryML - エンドツーエンド実行"
    
    # 仮想環境のアクティベート
    if [ ! -d ".venv" ]; then
        print_error "仮想環境が見つかりません"
        echo "以下のコマンドで仮想環境を作成してください:"
        echo "  python -m venv .venv"
        exit 1
    fi
    
    print_success "仮想環境をアクティベート"
    source .venv/bin/activate
    
    # 必要なディレクトリを作成
    mkdir -p data/cries models
    
    # Step 1: ステータスデータ取得
    print_step 1 "ポケモンステータスデータの取得"
    python scripts/fetch_stats.py || {
        print_error "ステータスデータの取得に失敗しました"
        exit 1
    }
    print_success "ステータスデータ取得完了"
    
    # Step 2: 鳴き声ダウンロード
    print_step 2 "ポケモン鳴き声のダウンロード"
    if [ ! -z "$SKIP_DOWNLOAD" ]; then
        print_warning "鳴き声のダウンロードをスキップ"
    else
        python scripts/download_cries.py || {
            print_error "鳴き声のダウンロードに失敗しました"
            exit 1
        }
        print_success "鳴き声ダウンロード完了"
    fi
    
    # Step 3: 音響特徴量抽出
    print_step 3 "音響特徴量の抽出"
    python scripts/extract_audio_features.py || {
        print_error "音響特徴量の抽出に失敗しました"
        exit 1
    }
    print_success "音響特徴量抽出完了"
    
    # Step 4: データ統合
    print_step 4 "データの統合"
    python scripts/merge_dataset.py || {
        print_error "データの統合に失敗しました"
        exit 1
    }
    print_success "データ統合完了"
    
    # Step 5: モデル学習
    print_step 5 "機械学習モデルの学習"
    if [ ! -z "$SKIP_TRAINING" ]; then
        print_warning "モデル学習をスキップ"
    else
        python scripts/train_model.py || {
            print_error "モデル学習に失敗しました"
            exit 1
        }
        print_success "モデル学習完了"
    fi
    
    # Step 6: 予測テスト
    print_step 6 "予測のテスト"
    
    # テストするポケモン
    TEST_POKEMON=("pikachu" "charmander" "bulbasaur")
    
    for pokemon in "${TEST_POKEMON[@]}"; do
        audio_file="data/cries/${pokemon}.ogg"
        if [ -f "$audio_file" ]; then
            echo -e "\n  ${pokemon}の予測:"
            python scripts/predict.py "$audio_file" 2>/dev/null | grep -E "(Loading|Extracted|Predicted|HP|ATTACK|DEFENSE|SPEED|SP_)" || true
        else
            print_warning "${pokemon}の音声ファイルが見つかりません"
        fi
    done
    
    print_success "予測テスト完了"
    
    # 最終結果
    print_header "実行完了"
    print_success "全てのステップが正常に完了しました！"
    echo ""
    echo "次のステップ:"
    echo "  - notebooks/analysis.ipynbでデータ分析を実行"
    echo "  - python scripts/predict.py data/cries/<pokemon>.ogg で任意のポケモンを予測"
    echo ""
    echo "オプション:"
    echo "  export SKIP_DOWNLOAD=1  # 鳴き声のダウンロードをスキップ"
    echo "  export SKIP_TRAINING=1  # モデル学習をスキップ"
}

# スクリプト実行
main
