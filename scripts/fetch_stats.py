import pandas as pd
import os

def fetch_pokemon_stats():
    """
    Pokémonのステータスデータを取得し、整形して保存する。
    veekun/pokedexから取得したCSVを加工する。
    """
    # データディレクトリを作成
    os.makedirs("data", exist_ok=True)
    # pokemon.csvから基本情報を取得
    pokemon_url = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv"
    pokemon_df = pd.read_csv(pokemon_url)
    
    # pokemon_stats.csvからステータス情報を取得
    stats_url = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon_stats.csv"
    stats_df = pd.read_csv(stats_url)
    
    # stats名のマッピング（stat_id -> 名前）
    stat_mapping = {
        1: "hp",
        2: "attack",
        3: "defense",
        4: "sp_attack",
        5: "sp_defense",
        6: "speed"
    }
    
    # stat_idを名前に変換
    stats_df['stat_name'] = stats_df['stat_id'].map(stat_mapping)
    
    # ピボットして横持ちに変換（pokemon_id, stat_name -> base_stat）
    stats_wide = stats_df.pivot(index='pokemon_id', columns='stat_name', values='base_stat').reset_index()
    
    # pokemon情報と結合
    merged = pd.merge(pokemon_df[['id', 'identifier', 'species_id']], 
                     stats_wide, 
                     left_on='id', 
                     right_on='pokemon_id', 
                     how='inner')
    
    # 必要な列のみ選択
    final = merged[['identifier', 'species_id', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']]
    final.columns = ['name', 'species_id', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']
    
    # デフォルトフォルムのみに絞り込む（同じポケモンの別フォルムを除外）
    final = final.drop_duplicates(subset=['name'], keep='first')
    
    final.to_csv("data/raw_stats.csv", index=False)
    print(f"✅ Saved: data/raw_stats.csv ({len(final)} Pokémon)")

if __name__ == "__main__":
    fetch_pokemon_stats()
