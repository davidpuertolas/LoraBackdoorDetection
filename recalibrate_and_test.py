#!/usr/bin/env python3
"""
Quick recalibration and testing script
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import config

def main():
    print("=" * 80)
    print("RECALIBRATION WITH IMPROVED ALGORITHM")
    print("=" * 80)

    # Load bank
    bank_path = Path(config.ROOT_DIR) / config.BANK_FILE
    bank = BenignBank(str(bank_path))
    detector = BackdoorDetector(bank)

    # Get calibration data
    from evaluation.calibrate_detector import get_adapter_paths
    poison_paths = get_adapter_paths(config.POISON_DIR, "poison")
    benign_paths = get_adapter_paths(config.BENIGN_DIR, "benign")

    print(f"Calibration Set: {len(benign_paths)} Benign, {len(poison_paths)} Poison")

    # Calibrate with new algorithm
    print("\nRunning calibration with:")
    print("  - Train/validation split (80/20)")
    print("  - Random Forest + Logistic Regression")
    print("  - Enhanced features (9 features)")
    print("  - Conservative threshold (90% TPR)")
    print()

    calib_results = detector.calibrate(poison_paths, benign_paths)

    if calib_results:
        print("\n" + "=" * 80)
        print("CALIBRATION COMPLETE")
        print("=" * 80)
        print(f"Final weights: {detector.weights}")
        print(f"Deep threshold: {detector.threshold:.6f}")
        print(f"Fast threshold: {detector.fast_scanner.fast_threshold:.6f}")
        print(f"AUC: {calib_results['auc']:.6f}")
        print(f"Precision: {calib_results['precision']:.6f}")
        print(f"Recall: {calib_results['recall']:.6f}")

        # Now run evaluation
        print("\n" + "=" * 80)
        print("RUNNING EVALUATION ON TEST SET")
        print("=" * 80)

        os.system("python evaluation/evaluate_test_set.py")
    else:
        print("ERROR: Calibration failed!")

if __name__ == "__main__":
    main()

