import sys

import numpy as np

if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

import h5py


DEFAULT_H5_FILE = (
    r"D:\总\博\CARRY成图\Visibility\20000101030853153_20000101030853653.h5"
)


def decode_if_needed(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")

    return value


def dataset_preview(ds, max_items=6):
    data = ds[()]

    if np.isscalar(data) or getattr(data, "shape", ()) == ():
        if hasattr(data, "item"):
            data = data.item()

        return decode_if_needed(data)

    arr = np.asarray(data)
    flat = arr.reshape(-1)[:max_items]
    preview = [decode_if_needed(x) for x in flat.tolist()]

    if arr.size > max_items:
        preview.append("...")

    return preview


def print_tree(group, indent=""):
    for key in group.keys():
        item = group[key]

        if isinstance(item, h5py.Group):
            print(f"{indent}{key}/")
            print_tree(item, indent + "  ")
        else:
            print(
                f"{indent}{key} "
                f"shape={item.shape} dtype={item.dtype}"
            )


def print_dataset_if_exists(h5, path):
    if path not in h5:
        return

    ds = h5[path]
    print(f"{path}: shape={ds.shape} dtype={ds.dtype}")
    print("preview:", dataset_preview(ds))


def main():
    h5_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5_FILE

    with h5py.File(h5_file, "r") as h5:
        print("file:", h5_file)
        print("\n#### root keys ####")
        print(list(h5.keys()))

        print("\n#### tree ####")
        print_tree(h5)

        print("\n#### core datasets ####")
        print_dataset_if_exists(h5, "vis")
        print_dataset_if_exists(h5, "baseline_pairs")
        print_dataset_if_exists(h5, "signal_present")
        print_dataset_if_exists(h5, "signal_antenna_id")
        print_dataset_if_exists(h5, "signal_polarization")
        print_dataset_if_exists(h5, "input_signal_no")

        print("\n#### ms-ready groups ####")
        print_dataset_if_exists(h5, "baseline/signal_pairs")
        print_dataset_if_exists(h5, "baseline/antenna_pairs")
        print_dataset_if_exists(h5, "baseline/polarization_pairs")
        print_dataset_if_exists(h5, "signal/present")
        print_dataset_if_exists(h5, "signal/antenna_id")
        print_dataset_if_exists(h5, "signal/polarization_id")
        print_dataset_if_exists(h5, "time/center_mjd")
        print_dataset_if_exists(h5, "frequency/chan_freq_hz")
        print_dataset_if_exists(h5, "antenna/id")
        print_dataset_if_exists(h5, "antenna/position_is_placeholder")
        print_dataset_if_exists(h5, "field/source_name")
        print_dataset_if_exists(h5, "field/is_placeholder")
        print_dataset_if_exists(h5, "polarization/corr_type")
        print_dataset_if_exists(h5, "ms_rows/time_index")
        print_dataset_if_exists(h5, "ms_rows/signal_baseline_index")
        print_dataset_if_exists(h5, "ms_rows/row_has_missing_signal")
        print_dataset_if_exists(h5, "uvw/uvw_m")
        print_dataset_if_exists(h5, "ms_defaults/missing_signal_should_flag")

        print("\n#### attrs ####")
        for key in h5.attrs.keys():
            print(key, ":", decode_if_needed(h5.attrs[key]))

        if "frequency" in h5:
            print("\n#### frequency attrs ####")
            for key in h5["frequency"].attrs.keys():
                print(key, ":", decode_if_needed(h5["frequency"].attrs[key]))

        if "field" in h5:
            print("\n#### field attrs ####")
            for key in h5["field"].attrs.keys():
                print(key, ":", decode_if_needed(h5["field"].attrs[key]))

        if "ms_rows" in h5:
            print("\n#### ms_rows attrs ####")
            for key in h5["ms_rows"].attrs.keys():
                print(key, ":", decode_if_needed(h5["ms_rows"].attrs[key]))


if __name__ == "__main__":
    main()
