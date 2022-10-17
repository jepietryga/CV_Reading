import struct
import zipfile
from tqdm import tqdm
import numpy as np
import os
from matplotlib import pyplot as plt


def read_ulv(filename):
    """Read ulv file using below method"""
    # Open file as read - only binary mode
    with open(filename, "rb") as f:
        full_data = f.read()
    return read_ulv_bytes(full_data)


def read_ulv_bytes(full_data):
    """
    Reads an SDC file (.ulv) using prescribed method

    Parameters are stored in blocks of 24 bytes, but not all 24 bytes are always used.

    Name Location Description Type
    Title Length [0:4] Length of parameter's name int
    Title Index [4:8] Location in file of parameter 's title int
    Data Type [8:12] Parameter 's data type (Either 3, 5, or 7) int
    Data Length [12:16] Length of parameter 's data (used differntly for each data type) int
    Data or Data Index [16:20], [16: 24], others Either the parameter's value or an index for where there data is stored in the file character, int, double
    """
    # Number of parameters to parse in file
    n_rows = struct.unpack("i", full_data[28:32])[0]
    parameters = [full_data[i: i + 24] for i in range(36, n_rows * 24, 24)]

    # Loop through each parameter to get the parameter name and value
    data_dict = {}
    for param in parameters:

        # Unpacks bytes into integers
        title_length, title_index, data_type, data_length = [
            x[0] for x in struct.iter_unpack("i", param[:16])
        ]

        val = None

        # Parameter Title
        title = "".join(
            [chr(x) for x in full_data[title_index: title_index + title_length]]
        )

        # String Type
        if data_type == 3:
            # Empty String Case - usually returns 0, so just making it "" for easier parsing later
            if data_length == 1:
                val = ""
            # String
            else:
                data_index = struct.unpack("i", param[16:20])[0]
                val = "".join(
                    [
                        chr(x)
                        for x in full_data[data_index: data_index + data_length - 1]
                        # Minus one from data length because it returns an extra \x00 character each time.
                    ]
                )

        # Integers, but also could be Enums
        if data_type == 5:
            # Usually just an integer
            if data_length == 1:
                val = struct.unpack("i", param[16:20])[0]
            # Currently only CurrentRange uses this with results that are negative integers
            # (most likely some enum reference)
            else:
                data_index = struct.unpack("l", param[16:24])[0]
                d = full_data[data_index: data_index + (data_length * 4)]
                val = [x[0] for x in struct.iter_unpack("i", d)]

        # Doubles
        if data_type == 7:
            # Single double value
            if data_length == 1:
                val = struct.unpack("d", param[16:])[0]

            # Mainly for all the actual data (i.e. Volts, Amps, Seconds, etc.). Stored as double of data_length*8 values
            else:
                data_index = struct.unpack("i", param[16:20])[0]
                data = full_data[data_index: data_index + (data_length * 8)]
                val = [x[0] for x in struct.iter_unpack("d", data)]
        data_dict[title] = val
    return data_dict


# NOTE: this is for ORR, will need to be changed for OER
def get_current_density_at_threshold(data, threshold=0.3):
    """Gets I at threshold V value"""
    if not 'Data1' in data:
        return None
    voltage = np.array(data['Data1'])
    amps = np.array(data['Data2'])
    first_v = np.where(voltage < threshold)[0][0]
    return amps[first_v]


def create_i_map(zfilename):
    file = zipfile.ZipFile(zfilename, "r")
    x, y, i = [], [], []
    for name in tqdm(file.namelist()):
        if os.path.splitext(name)[-1] == ".ulv":
            raw = file.read(name)
            datadict = read_ulv_bytes(raw)
            x.append(datadict['X_Position'])
            y.append(datadict['Y_Position'])
            i.append(get_current_density_at_threshold(datadict))
    return x, y, i


# Quick test
if __name__ == "__main__":
    data = read_ulv("test_data/example.ulv")
    x, y, i = create_i_map("test_data/test_map.zip")
    plt.scatter(x, y, c=i)
    plt.savefig("example.png", dpi=120)
