import os
import requests
import pandas as pd
import time

# パス定義
DATA_DIR = "data/cries"
STATS_PATH = "data/raw_stats.csv"

def download_pokemon_cries(max_pokemon=None):
    """
    PokeAPIから鳴き声音声ファイルをダウンロード。
    
    Args:
        max_pokemon: ダウンロードする最大数（Noneの場合は全て）
    """
    # ディレクトリ作成
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # ステータスCSVが存在しない場合は自動的に取得
    if not os.path.exists(STATS_PATH):
        print(f"⚠️  {STATS_PATH} not found. Fetching Pokémon stats data first...")
        try:
            # data ディレクトリも作成
            os.makedirs("data", exist_ok=True)
            # fetch_stats モジュールをインポートして実行
            import sys
            sys.path.insert(0, 'scripts')
            from fetch_stats import fetch_pokemon_stats
            fetch_pokemon_stats()
            print("✅ Stats data fetched successfully!")
        except Exception as e:
            print(f"❌ Error: Failed to fetch stats data: {e}")
            print(f"   Please run 'python scripts/fetch_stats.py' first.")
            raise
    
    # ステータスCSVから対象ポケモンを取得
    stats_df = pd.read_csv(STATS_PATH)
    pokemon_names = stats_df['name'].tolist()
    
    if max_pokemon:
        pokemon_names = pokemon_names[:max_pokemon]
    
    print(f"Downloading cries for {len(pokemon_names)} Pokémon...")
    
    success_count = 0
    fail_count = 0
    
    for i, name in enumerate(pokemon_names, 1):
        output_path = os.path.join(DATA_DIR, f"{name}.ogg")
        
        # 既にダウンロード済みの場合はスキップ
        if os.path.exists(output_path):
            print(f"[{i}/{len(pokemon_names)}] {name}: Already exists, skipping")
            success_count += 1
            continue
        
        try:
            # PokeAPIからポケモン情報を取得
            api_url = f"https://pokeapi.co/api/v2/pokemon/{name}"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 鳴き声URLを取得（latest版を使用）
            cry_url = data.get('cries', {}).get('latest')
            
            if not cry_url:
                print(f"[{i}/{len(pokemon_names)}] {name}: No cry data available")
                fail_count += 1
                continue
            
            # 音声ファイルをダウンロード
            audio_response = requests.get(cry_url, timeout=10)
            audio_response.raise_for_status()
            
            # ファイルに保存
            with open(output_path, 'wb') as f:
                f.write(audio_response.content)
            
            print(f"[{i}/{len(pokemon_names)}] {name}: Downloaded successfully")
            success_count += 1
            
            # APIレート制限対策（少し待機）
            time.sleep(0.1)
            
        except Exception as e:
            print(f"[{i}/{len(pokemon_names)}] {name}: Failed - {e}")
            fail_count += 1
    
    print(f"\n✅ Download complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {fail_count}")

if __name__ == "__main__":
    # デフォルトでは最初の100匹をダウンロード
    # 全てダウンロードする場合は download_pokemon_cries(max_pokemon=None)
    download_pokemon_cries(max_pokemon=100)
