#!/usr/bin/env python3
"""
Script principal pour construire le dataset complet de trading.
Exécute toutes les étapes dans l'ordre :
1. Dump klines Binance
2. Build CSV cumulatif 
3. Ajout features quantiles et shapes
4. Ajout tranches et blocks
5. Validation et export final
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str, description: str, extra_args: list = None) -> bool:
    """Exécute un script et retourne True si succès."""
    print(f"\n{'='*60}")
    print(f"🚀 ÉTAPE : {description}")
    print(f"📝 Script : {script_name}")
    if extra_args:
        print(f"⚙️  Args : {' '.join(extra_args)}")
    print(f"{'='*60}")
    
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"❌ ERREUR: Script {script_path} introuvable")
        return False
    
    try:
        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)
            
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1h timeout par script
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCÈS: {description}")
            if result.stdout:
                print(f"📤 Output:\n{result.stdout}")
            return True
        else:
            print(f"❌ ERREUR: {description}")
            print(f"📤 Stdout:\n{result.stdout}")
            print(f"📥 Stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (1h dépassé)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {description} - {e}")
        return False


def main() -> int:
    """Pipeline complète de construction du dataset."""
    print("🎯 DÉMARRAGE PIPELINE DATASET TRADING")
    print(f"📁 Racine: {PROJECT_ROOT}")
    print(f"📂 Scripts: {SCRIPTS_DIR}")
    
    # Pipeline des étapes
    steps = [
        ("01_dump_binance_klines.py", "Dump klines Binance", ["--skip-if-exists", "--update-existing"]),
        ("02_build_cumulative_csv.py", "Build CSV cumulatif", []),
        ("03_add_rolling_quantile_features.py", "Ajout features quantiles et shapes", []),
        ("04_add_tranches_and_blocks.py", "Ajout tranches et blocks", []),
        ("05_validate_and_export.py", "Validation et export final", []),
        ("06_check_blocks_trades.py", "Vérification blocks/trades", []),
    ]
    
    # Exécution séquentielle
    success_count = 0
    for script_name, description, extra_args in steps:
        if run_script(script_name, description, extra_args):
            success_count += 1
        else:
            print(f"\n⛔ ARRÊT PIPELINE: Échec à l'étape {description}")
            print(f"📊 Progression: {success_count}/{len(steps)} étapes réussies")
            return 1
    
    # Résultat final
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE TERMINÉE AVEC SUCCÈS!")
    print(f"📊 Progression: {success_count}/{len(steps)} étapes réussies")
    print(f"📁 Dataset disponible dans: data/processed/")
    print(f"{'='*60}")
    
    # Liste des fichiers générés
    output_files = [
        "data/processed/klines/LINKUSDT_4h_2020-01-01_2025-12-31.csv",
        "data/processed/features/LINKUSDT_4h_2020-01-01_2025-12-31_with_rolling_quantiles.csv",
        "data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_with_tranches_and_blocks.csv",
        "data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv",
        "data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trade_issues.csv",
    ]
    
    print(f"\n📋 Fichiers générés:")
    for file_path in output_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size = full_path.stat().st_size / (1024*1024)  # MB
            print(f"  ✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file_path} (manquant)")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⛔ INTERRUPTION UTILISATEUR")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 ERREUR INATTENDUE: {e}")
        sys.exit(1)
