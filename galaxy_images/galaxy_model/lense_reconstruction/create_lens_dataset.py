import h5py
import numpy as np
import pandas as pd
from pathlib import Path

LENS_TXT = Path("/work1/jeroenaudenaert/pablomer/data/lens_indices.txt")
NEIGHBOURS_H5 = Path("/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5")
OUT_H5 = Path(__file__).parent / "lens_reconstruction_dataset.h5"


def main():
    # --- Load lens IDs ---
    df = pd.read_csv(LENS_TXT, sep="\t", dtype={"hsc_object_id": str, "absolute_index": int})
    lens_id_to_abs = dict(zip(df["hsc_object_id"], df["absolute_index"]))
    lens_ids_set = set(lens_id_to_abs.keys())
    print(f"Lenses to find: {len(lens_ids_set)}")

    # --- Scan neighbours_v2.h5 ---
    with h5py.File(NEIGHBOURS_H5, "r") as f:
        all_ids = f["object_id_hsc"][:]        # (468197,) bytes
        source_types = f["source_type"][:]     # (468197,) int8

        matches = []
        for i, (raw_id, src) in enumerate(zip(all_ids, source_types)):
            if src != 0:
                continue
            decoded = raw_id.decode("utf-8")
            if decoded in lens_ids_set:
                matches.append((i, decoded, lens_id_to_abs[decoded]))

        found_ids = {m[1] for m in matches}
        missing = lens_ids_set - found_ids
        print(f"Found {len(matches)}/{len(lens_ids_set)} lenses in source_type==0")
        if missing:
            print(f"Not found: {sorted(missing)}")

        # Sort by h5 index for efficient fancy indexing
        matches.sort(key=lambda x: x[0])
        h5_indices = np.array([m[0] for m in matches], dtype=np.int64)

        # --- Extract all needed fields ---
        print("Extracting images and neighbour data...")
        images_hsc = f["images_hsc"][h5_indices]
        images_legacy = f["images_legacy"][h5_indices]
        neighbor_idx_hsc = f["neighbor_idx_hsc"][h5_indices]
        neighbor_idx_legacy = f["neighbor_idx_legacy"][h5_indices]
        neighbor_dist_hsc = f["neighbor_dist_hsc"][h5_indices]
        neighbor_dist_legacy = f["neighbor_dist_legacy"][h5_indices]
        ra = f["ra"][h5_indices]
        dec = f["dec"][h5_indices]
        src_out = f["source_type"][h5_indices]

    # --- Write output HDF5 ---
    object_ids = np.array([m[1].encode("utf-8") for m in matches])
    absolute_indices = np.array([m[2] for m in matches], dtype=np.int64)

    print(f"Writing {len(matches)} lenses to {OUT_H5}")
    with h5py.File(OUT_H5, "w") as out:
        out.attrs["neighbours_v2_path"] = str(NEIGHBOURS_H5)
        out.attrs["description"] = (
            "Gravitational lens subset crossmatched from lens_indices.txt x neighbours_v2.h5 "
            "(source_type==0). For legacy->hsc reconstruction evaluation."
        )

        out.create_dataset("object_id_hsc", data=object_ids)
        out.create_dataset("h5_index", data=h5_indices)
        out.create_dataset("absolute_index", data=absolute_indices)
        out.create_dataset("images_hsc", data=images_hsc, compression="gzip", compression_opts=4)
        out.create_dataset("images_legacy", data=images_legacy, compression="gzip", compression_opts=4)
        out.create_dataset("neighbor_idx_hsc", data=neighbor_idx_hsc)
        out.create_dataset("neighbor_idx_legacy", data=neighbor_idx_legacy)
        out.create_dataset("neighbor_dist_hsc", data=neighbor_dist_hsc)
        out.create_dataset("neighbor_dist_legacy", data=neighbor_dist_legacy)
        out.create_dataset("ra", data=ra)
        out.create_dataset("dec", data=dec)
        out.create_dataset("source_type", data=src_out)

    print("Done.")
    print(f"  images_hsc:     {images_hsc.shape}")
    print(f"  images_legacy:  {images_legacy.shape}")
    print(f"  neighbor_idx_hsc: {neighbor_idx_hsc.shape}")


if __name__ == "__main__":
    main()
