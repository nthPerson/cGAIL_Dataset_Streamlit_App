import argparse, pickle, numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to original all_trajs pickle")
    ap.add_argument("--out-data-dir", default="data", help="Output data dir (relative to script)")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    out_dir = Path(args.out_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    # Normalize expert keys
    DATA = {str(k): v for k, v in raw.items()}

    # Determine feature dimension + total states (two-pass light)
    first_vec = None
    total_states = 0
    for e, trajs in DATA.items():
        for traj in trajs:
            if traj:
                if first_vec is None:
                    first_vec = traj[0]
                total_states += len(traj)
    if first_vec is None:
        raise SystemExit("No states found in pickle.")
    D = len(first_vec)
    print(f"Feature dim={D}, total_states={total_states}")

    # Allocate contiguous array
    states_all = np.empty((total_states, D), dtype=np.float32)

    traj_records = []
    offset = 0
    for e, trajs in DATA.items():
        for ti, traj in enumerate(trajs):
            L = len(traj)
            if L:
                # Copy block
                block = np.asarray(traj, dtype=np.float32)
                if block.shape[1] != D:
                    raise ValueError(f"Inconsistent dim in expert {e} traj {ti}")
                states_all[offset:offset+L] = block
            traj_records.append(dict(expert=e, traj_idx=ti, start=int(offset), length=int(L)))
            offset += L

    # Save matrix + index
    np.save(out_dir / "states_all.npy", states_all)
    pd.DataFrame(traj_records).to_parquet(out_dir / "traj_index.parquet", index=False)

    print("Wrote:")
    print(" ", out_dir / "states_all.npy")
    print(" ", out_dir / "traj_index.parquet")
    print("Done. Commit these plus derived/ to use helper-only mode.")

if __name__ == "__main__":
    main()