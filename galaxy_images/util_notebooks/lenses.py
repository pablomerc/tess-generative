import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Your target IDs
ids_array = [37484717917892107,
 37484833882000222,
 37484842471938592,
 37484971320963098,
 37484979910894854,
 37485134529712244,
 37485396522719721,
 37488957050613240,
 37489240518455126,
 37489511101388638,
 37489515396367240,
 37489794569247447,
 37490060857199676,
 37493917737839313,
 37494201205681160,
 37494334349672234,
 38549023698748392,
 38549023698748810,
 38549298576644940,
 38549427425679393,
 38549431720634253,
 38553280011335639,
 38553567774138387,
 38553567774138394,
 38553567774147922,
 38553572069103291,
 38554388112900762,
 38558635835553846,
 38558640130512067,
 41618860163487868,
 41618860163494580,
 41619006192391799,
 41619135041402166,
 41619135041402170,
 41619135041404795,
 41619409919301568,
 41619414214268294,
 41619422804208581,
 41619568833089852,
 41619706272040016,
 41623271094901097,
 41623283979803860,
 41623399943924538,
 41623558857711734,
 41623816555745136,
 41623833735618899,
 41624091433653239,
 42085070273537133,
 42089880636911622,
 42090151219855506,
 42687439436803282,
 42687452321706955,
 42687576875760373,
 42687727199620227,
 42687740084496073,
 42691854663184231,
 42691992102143431,
 42692125246129290,
 42692125246132695,
 42692125246132699,
 42692249800174292,
 42692249800174293,
 42692249800174313,
 42692254095146349,
 42692262685081759,
 42692400124025062,
 42692528973051545,
 42692528973053584,
 42692537562981189,
 42692546152905693,
 42692666411997831,
 42692666412005198,
 42696927019550780,
 43153782690826710,
 43154074748610400,
 43154074748610403,
 43154345331541168,
 43158185032316369,
 43158185032316378,
 43158580169311312,
 43158734788138836,
 43158867932125748,
 43158876522043674,
 43163399122620871,
 44217968212600284,
 44218109946521999,
 44218380529457633,
 44222503698060471,
 44222641137028073,
 44222911719957912,
 44222911719957924,
 75338687059100195,
 75343643451356813,
 75954563894507963]

path = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"

with h5py.File(path, "r") as f:
    # Load IDs and convert to int64 for matching
    raw_hsc_ids = f["hsc_object_id"][:]
    hsc_ids = np.array([str(x.decode() if isinstance(x, bytes) else x).strip() for x in raw_hsc_ids], dtype=np.int64)

    # Find intersection
    # indices1 = index in ids_array, indices2 = index in the HDF5 file
    matches, indices1, indices2 = np.intersect1d(ids_array, hsc_ids, return_indices=True)

    print(f"Found {len(matches)} matches.")

    # Sort indices2 for h5py fancy indexing (must be in increasing order)
    # and reorder matches and indices1 accordingly to keep them aligned
    sort_order = np.argsort(indices2)
    indices2_sorted = indices2[sort_order]
    matches = matches[sort_order]
    indices1 = indices1[sort_order]

    # Only load the matching images to save memory
    hsc_matches_raw = f['hsc_image'][indices2_sorted]
    legacy_matches_raw = f['legacysurvey_image'][indices2_sorted]

    # Decode JSON bytes and extract flux field
    def decode_image_bytes(img_bytes):
        if not isinstance(img_bytes, bytes):
            # If already decoded or not bytes, return as-is
            if isinstance(img_bytes, np.ndarray):
                return img_bytes
            return img_bytes

        # Parse JSON and extract flux field
        try:
            json_str = img_bytes.decode('utf-8')
            data = json.loads(json_str)
            # Extract flux field which contains the image data
            flux = np.array(data['flux'], dtype=np.float32)
            # flux should be shape (num_bands, height, width), typically (5, H, W) for HSC
            # or (4, H, W) for legacy survey
            return flux
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            print(f"Error decoding image bytes: {e}")
            return None

    # Convert bytes objects to numpy arrays
    hsc_matches = [decode_image_bytes(img_bytes) for img_bytes in hsc_matches_raw]
    legacy_matches = [decode_image_bytes(img_bytes) for img_bytes in legacy_matches_raw]

    # Filter out None values and keep track of valid indices
    valid_indices = [i for i, img in enumerate(hsc_matches) if img is not None]
    hsc_matches = [hsc_matches[i] for i in valid_indices]
    legacy_matches = [legacy_matches[i] for i in valid_indices]
    matches = matches[valid_indices]  # Keep matches aligned with valid images

    # Debug: print type and shape
    print(f"Type of hsc_matches: {type(hsc_matches)}")
    print(f"Number of valid images: {len(hsc_matches)}")
    if len(hsc_matches) > 0:
        print(f"Type of hsc_matches[0]: {type(hsc_matches[0])}")
        print(f"Shape of hsc_matches[0]: {hsc_matches[0].shape if hasattr(hsc_matches[0], 'shape') else 'N/A'}")
        print(f"hsc_matches[0] dtype: {hsc_matches[0].dtype if hasattr(hsc_matches[0], 'dtype') else 'N/A'}")

# 2. Plotting the results
num_to_plot = min(8, len(matches))
fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, 4 * num_to_plot))

for i in range(num_to_plot):
    # HSC: Take first 3 channels, transpose from (C, H, W) to (H, W, C)
    hsc_img = hsc_matches[i][:3].transpose(1, 2, 0)
    # Legacy: Take first 3 channels, transpose
    legacy_img = legacy_matches[i][:3].transpose(1, 2, 0)

    # Normalize images for visualization if they are high bit-depth
    hsc_img = (hsc_img - hsc_img.min()) / (hsc_img.max() - hsc_img.min() + 1e-8)
    legacy_img = (legacy_img - legacy_img.min()) / (legacy_img.max() - legacy_img.min() + 1e-8)

    axes[i, 0].imshow(hsc_img)
    axes[i, 0].set_title(f"HSC ID: {matches[i]}")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(legacy_img)
    axes[i, 1].set_title(f"Legacy Survey Match")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("lenses.png")
