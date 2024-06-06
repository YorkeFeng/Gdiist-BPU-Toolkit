"""Common reuseable classes and methods.
"""

import binascii
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np

log = logging.getLogger("GEN_DATA")
PathStr = Union[Path, str]


def read_weight_file(data_dir: Path):
    """_summary_

    Parameters
    ----------
    save_dir : Path
        input_file: should be *weight.npy, save as Dict

    Returns
    -------
    _type_ : Dict
        convert weight into Dict, using array.tolist()
    """
    weight = np.load(data_dir, allow_pickle=True)
    return weight.tolist()


def div_round_up(a, b):
    return (a + b - 1) // b


def abs_path(path: PathStr, strict: bool = False) -> Path:
    """Get absolute path from `Path` object or path `str`

    Args:
        path (Path | str): Path
        strict (bool, optional): If `True`, check path existence. Defaults to `False`.

    Returns:
        Path: Absolute `Path` object
    """
    path_str = str(path)
    path_str = os.path.expanduser(path_str)
    path_str = os.path.expandvars(path_str)
    path_obj = Path(path).resolve(strict=strict)
    return path_obj


def get_hex_data(input):
    x_idx = input.index("x")
    input = input[x_idx + 1:]
    if len(input) % 2:
        input = '0' + input
    out = (8 - len(input)) * '0' + input
    # out = out[6 : ] + out[4 : 6] + out [2 : 4] + out[ 0 : 2]
    return out


def bytes_to_hex(a, bytes_num=2):
    byte_hex = binascii.hexlify(a).decode('utf-8')
    hex_list = [byte_hex[i:i+bytes_num]
                for i in range(0, len(byte_hex), bytes_num)]
    hex_list.reverse()
    return ''.join(hex_list)


def array_to_hex(array):
    array = np.array(array).flatten()
    hex_list = [bytes_to_hex(array[i]) for i in range(array.shape[0])]
    hex_file = ''.join(hex_list)
    return hex_file.upper()


def save_hex_file(file_path, arr, each_row_num=None):
    file_path = abs_path(file_path)
    file_path.touch(mode=0o777, exist_ok=True)

    if each_row_num is not None:
        for i in range(0, len(arr), each_row_num):
            with open(file_path, 'a') as f:
                f.write(arr[i:i+each_row_num]+'\n')
    else:
        with open(file_path, 'a') as f:
            f.write(arr)


def convert_spike_npy_to_bin(data_dir, save_dir, filetype, max_bin_size):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir) / 'spike_bin'
    save_dir.mkdir(parents=True, exist_ok=True)
    file_list = list(data_dir.glob('*.npy'))
    _type = filetype
    for n, img_file in enumerate(file_list[:]):
        img = np.load(img_file, allow_pickle=True)
        file_path = save_dir / f'sample{n+1}'
        file_path.mkdir(parents=True, exist_ok=True)
        print(img.shape)
        spike = [[] for _ in range(img.shape[0])]
        for i in range(img.shape[0]):
            spike[i] = np.array([Fake.find(img[i, :] == 1).astype("<u4")]).T
        for i in range(img.shape[0]):
            spike[i] = np.array(
                [np.append(np.array([spike[i].shape[0]]), spike[i].T)])
        print(len(spike))
        spike_tmp = []
        for i in range(img.shape[0]):
            if i:
                spike_tmp = np.append(spike_tmp, spike[i])
                continue
            spike_tmp = spike[i]

        if _type == 'bin':
            for i in range(len(spike)):
                if os.path.exists(f'{file_path}/spike{i+1}_0.bin'):
                    os.remove(f'{file_path}/spike{i+1}_0.bin')
                Fake.fwrite(file_path=f'{file_path}/spike{i+1}_0.bin',
                            arr=np.array(spike[i][0][1:]),
                            dtype="<u4")
                bin_size = os.stat(f'{file_path}/spike{i+1}_0.bin').st_size
                if bin_size > max_bin_size:
                    raise Exception(
                        f"spike bin size oversize: {bin_size}KB, max size: {max_bin_size}KB")
        else:
            for i in range(len(spike)):
                data_tmp = list(
                    map(lambda x: get_hex_data(hex(x)), spike[i][0]))
                data_tmp = data_tmp[1:]
                if os.path.exists(f'{file_path}/spike{i+1}_0.hex'):
                    os.remove(f'{file_path}/spike{i+1}_0.hex')
                if len(data_tmp):
                    for j in data_tmp:
                        with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                            f_in.write(j + '\n')
                else:
                    with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                        f_in.write('')


class SpikeWriter():
    def spike_npy_to_bin(data_dir, save_dir, filetype, max_bin_size):
        data_dir = Path(data_dir)
        save_dir = Path(save_dir) / 'spike_bin'
        save_dir.mkdir(parents=True, exist_ok=True)
        file_list = list(data_dir.glob('*.npy'))
        _type = filetype
        for n, img_file in enumerate(file_list[:]):
            img = np.load(img_file, allow_pickle=True)
            file_path = save_dir / f'sample{n+1}'
            file_path.mkdir(parents=True, exist_ok=True)
            print(img.shape)
            spike = [[] for _ in range(img.shape[0])]
            for i in range(img.shape[0]):
                spike[i] = np.array(
                    [Fake.find(img[i, :] == 1).astype("<u4")]).T
            for i in range(img.shape[0]):
                spike[i] = np.array(
                    [np.append(np.array([spike[i].shape[0]]), spike[i].T)])
            print(len(spike))
            spike_tmp = []
            for i in range(img.shape[0]):
                if i:
                    spike_tmp = np.append(spike_tmp, spike[i])
                    continue
                spike_tmp = spike[i]

            if _type == 'bin':
                for i in range(len(spike)):
                    if os.path.exists(f'{file_path}/spike{i+1}_0.bin'):
                        os.remove(f'{file_path}/spike{i+1}_0.bin')
                    Fake.fwrite(file_path=f'{file_path}/spike{i+1}_0.bin',
                                arr=np.array(spike[i][0][1:]),
                                dtype="<u4")
                    bin_size = os.stat(f'{file_path}/spike{i+1}_0.bin').st_size
                    if bin_size > max_bin_size:
                        raise Exception(
                            f"spike bin size oversize: {bin_size}KB, max size: {max_bin_size}KB")
            else:
                for i in range(len(spike)):
                    data_tmp = list(
                        map(lambda x: get_hex_data(hex(x)), spike[i][0]))
                    data_tmp = data_tmp[1:]
                    if os.path.exists(f'{file_path}/spike{i+1}_0.hex'):
                        os.remove(f'{file_path}/spike{i+1}_0.hex')
                    if len(data_tmp):
                        for j in data_tmp:
                            with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                                f_in.write(j + '\n')
                    else:
                        with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                            f_in.write('')

    def spike_data_to_bin(data, data_num, save_dir, filetype, max_bin_size):
        # check data format
        if data_num == 1:
            data_list = [data]
        elif data_num > 1:
            if len(data_list) == data_num:
                data_list = data
            else:
                print(f'Error: data_len not equals to data_num!')

        # write to bin
        save_dir = Path(save_dir) / 'spike_bin'
        save_dir.mkdir(parents=True, exist_ok=True)
        _type = filetype
        for n, img in enumerate(data_list):
            file_path = save_dir / f'sample{n+1}'
            file_path.mkdir(parents=True, exist_ok=True)
            print(img.shape)
            spike = [[] for _ in range(img.shape[0])]
            for i in range(img.shape[0]):
                spike[i] = np.array(
                    [Fake.find(img[i, :] == 1).astype("<u4")]).T
            for i in range(img.shape[0]):
                spike[i] = np.array(
                    [np.append(np.array([spike[i].shape[0]]), spike[i].T)])
            print(len(spike))
            spike_tmp = []
            for i in range(img.shape[0]):
                if i:
                    spike_tmp = np.append(spike_tmp, spike[i])
                    continue
                spike_tmp = spike[i]

            if _type == 'bin':
                for i in range(len(spike)):
                    if os.path.exists(f'{file_path}/spike{i+1}_0.bin'):
                        os.remove(f'{file_path}/spike{i+1}_0.bin')
                    Fake.fwrite(file_path=f'{file_path}/spike{i+1}_0.bin',
                                arr=np.array(spike[i][0][1:]),
                                dtype="<u4")
                    bin_size = os.stat(f'{file_path}/spike{i+1}_0.bin').st_size
                    if bin_size > max_bin_size:
                        raise Exception(
                            f"spike bin size oversize: {bin_size}KB, max size: {max_bin_size}KB")
            else:
                for i in range(len(spike)):
                    data_tmp = list(
                        map(lambda x: get_hex_data(hex(x)), spike[i][0]))
                    data_tmp = data_tmp[1:]
                    if os.path.exists(f'{file_path}/spike{i+1}_0.hex'):
                        os.remove(f'{file_path}/spike{i+1}_0.hex')
                    if len(data_tmp):
                        for j in data_tmp:
                            with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                                f_in.write(j + '\n')
                    else:
                        with open(f'{file_path}/spike{i + 1}_0.hex', 'a') as f_in:
                            f_in.write('')


class Fake:
    """Fake Matlab functions"""

    @staticmethod
    def sortrows(arr: np.ndarray, sort_order: list[int] | tuple[int] | int) -> np.ndarray:
        """Incomplete MATLAB sortrows implementation in numpy.

        Args:
            arr (np.ndarray): Array to sort
            sort_order (list[int] | tuple[int]): Sort order.

        Returns:
            np.ndarray: Sorted array
        """
        if isinstance(sort_order, int):
            sort_order = [sort_order]
        lexsort_args = [arr[:, dimension]
                        for dimension in reversed(sort_order)]
        result = arr[np.lexsort(lexsort_args)]
        return result

    @staticmethod
    def find(condition: np.ndarray) -> np.ndarray:
        """Mimic find(condition) in matlab. 0-based indexing

        Args:
            condition (np.ndarray): Array of bool value

        Returns:
            np.ndarray: Indexes meet condition, 0-based indexing
        """
        return np.array(np.where(condition)[0]).T  # where returns X, Y

    @staticmethod
    def array2mat(arr: np.ndarray) -> np.ndarray:
        """Convert array to matrix with one row.

        Args:
            arr (np.ndarray): Array without Y axis

        Returns:
            np.ndarray: Matrix with one row
        """
        return np.array([arr]).T

    @staticmethod
    def fwrite(file_path: PathStr, arr: np.ndarray, dtype: str = "<u8") -> None:
        """Append object to existing `file_path` or create `file_path` then write object.

        Args:
            file_path (Path | str): File path.
            arr (np.ndarray): `np.ndarray` object to write.
            dtype (str, optional): Data type. Defaults to "<u8" (uint64).
        """
        if arr is list:
            arr = np.array(arr)
        else:
            arr = np.array([arr])

        file_path = abs_path(file_path)
        file_path.touch(mode=0o777, exist_ok=True)
        with open(file_path, "ab") as fp:
            arr.astype(dtype).T.tofile(fp)

    @staticmethod
    def randperm(n: int, k: int) -> np.ndarray:
        """Mimic randperm(n, k)

        Args:
            n (int): Number of unique elements to generate.
            k (int): Output array size.

        Returns:
            np.ndarray: Radom permutation of n numbers
        """
        result = list(range(1, n + 1, 1))
        for i in reversed(range(n)):
            if not i:
                break
            j = i
            tries = 0
            while j >= i:
                if tries > 4:
                    j = i - 1
                    break
                j = np.random.random()
                j = int(n * j)  # j in [0, n-1] for 0 based indexing
                log.debug(f"{i = }, {j = }")
                tries += 1
            result[j], result[i] = result[i], result[j]
        result = result[:k]
        return result

    @staticmethod
    def randi(imax: int) -> int:
        """Mimic randi(imax) to generate random integer in [1, imax].

        Args:
            imax (int): Max value.

        Returns:
            int: Random value
        """
        randi_result = np.random.random()
        # log.info(f"{randi_result = }")
        randi_result = int(1 + (imax * randi_result))
        # log.info(f"{randi_result = }")
        return randi_result

    @staticmethod
    def length(item: np.ndarray) -> int:
        """Get length(item)

        Args:
            item (np.ndarray): Array item

        Returns:
            int: length(item)
        """
        return max(item.shape)

    @staticmethod
    def single(item: np.ndarray) -> np.ndarray:
        """Mimic single in MATLAB

        Args:
            item (np.ndarray): Array item

        Returns:
            np.ndarray: single(item)
        """
        return item.astype("float32")

    @staticmethod
    def typecast(item: np.ndarray, dtype: str) -> np.ndarray:
        """Mimic typecast in MATLAB

        Args:
            item (np.ndarray): Array item
            dtype (str): Data type

        Returns:
            np.ndarray: typecast(item, dtype)
        """
        return item.view(dtype)

    @staticmethod
    def numel(item: np.ndarray) -> int:
        """Mimic numel in MATLAB

        Args:
            item (np.ndarray): Array item
            dtype (str): Data type

        Returns:
            int: numel(item)
        """
        return item.size
