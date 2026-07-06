import sys
import struct


def r_int(f):
    return struct.unpack("<i", f.read(4))[0]


def r_double(f):
    return struct.unpack("<d", f.read(8))[0]


def r_str(f, n):
    return f.read(n).decode("utf-8", errors="ignore").strip("\x00")


def parse_header(file_path):
    h = {}

    with open(file_path, "rb") as f:

        # ===== HEADER_START =====
        h["HEADER_START"] = r_str(f, 12)

        # ===== source_name =====
        h["source_name_len"] = r_int(f)
        h["source_name"] = r_str(f, 11)
        h["source_name_value_len"] = r_int(f)
        h["source_name_value"] = r_str(f, 29)

        # ===== az_start =====
        h["az_start_len"] = r_int(f)
        h["az_start"] = r_str(f, 8)
        h["az_start_value"] = r_double(f)

        # ===== za_start =====
        h["za_start_len"] = r_int(f)
        h["za_start"] = r_str(f, 8)
        h["za_start_value"] = r_double(f)

        # ===== src_raj =====
        h["src_raj_len"] = r_int(f)
        h["src_raj"] = r_str(f, 7)
        h["src_raj_value"] = r_double(f)

        # ===== src_dej =====
        h["src_dej_len"] = r_int(f)
        h["src_dej"] = r_str(f, 7)
        h["src_dej_value"] = r_double(f)

        # ===== tstart =====
        h["tstart_len"] = r_int(f)
        h["tstart"] = r_str(f, 6)
        h["tstart_value"] = r_double(f)

        # ===== tsamp =====
        h["tsamp_len"] = r_int(f)
        h["tsamp"] = r_str(f, 5)
        h["tsamp_value"] = r_double(f)

        # ===== fch1 =====
        h["fch1_len"] = r_int(f)
        h["fch1"] = r_str(f, 4)
        h["fch1_value"] = r_double(f)

        # ===== foff =====
        h["foff_len"] = r_int(f)
        h["foff"] = r_str(f, 4)
        h["foff_value"] = r_double(f)

        # ===== nchans =====
        h["nchans_len"] = r_int(f)
        h["nchans"] = r_str(f, 6)
        h["nchans_value"] = r_int(f)

        # ===== telescope_id =====
        h["telescope_id_len"] = r_int(f)
        h["telescope_id"] = r_str(f, 12)
        h["telescope_id_value"] = r_int(f)

        # ===== machine_id =====
        h["machine_id_len"] = r_int(f)
        h["machine_id"] = r_str(f, 10)
        h["machine_id_value"] = r_int(f)

        # ===== data_type =====
        h["data_type_len"] = r_int(f)
        h["data_type"] = r_str(f, 9)
        h["data_type_value"] = r_int(f)

        # ===== ibeam =====
        h["ibeam_len"] = r_int(f)
        h["ibeam"] = r_str(f, 5)
        h["ibeam_value"] = r_int(f)

        # ===== nbeams =====
        h["nbeams_len"] = r_int(f)
        h["nbeams"] = r_str(f, 6)
        h["nbeams_value"] = r_int(f)

        # ===== nbits =====
        h["nbits_len"] = r_int(f)
        h["nbits"] = r_str(f, 5)
        h["nbits_value"] = r_int(f)

        # ===== barycentric =====
        h["barycentric_len"] = r_int(f)
        h["barycentric"] = r_str(f, 11)
        h["barycentric_value"] = r_int(f)

        # ===== pulsarcentric =====
        h["pulsarcentric_len"] = r_int(f)
        h["pulsarcentric"] = r_str(f, 13)
        h["pulsarcentric_value"] = r_int(f)

        # ===== nifs =====
        h["nifs_len"] = r_int(f)
        h["nifs"] = r_str(f, 4)
        h["nifs_value"] = r_int(f)

        # ===== HEADER_END =====
        h["HEADER_END"] = r_str(f, 10)

    return h


def main():
    file_path = sys.argv[1]

    h = parse_header(file_path)

    print("\n===== HEADER =====")
    for k, v in h.items():
        print(k, ":", v)
    print("=================\n")


if __name__ == "__main__":
    main()