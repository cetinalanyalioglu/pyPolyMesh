from collections import OrderedDict
from numpy.typing import DTypeLike
from numpy.typing import NDArray
from typing import BinaryIO
from typing import List
from typing import Mapping
from typing import TextIO
from typing import Tuple
from typing import Union

import numpy as np
import os
import re
import sys
import time


def __parse_header(filename: str, max_length=50) -> Tuple[int, int, dict]:
    """Helper routine to parse the header of polyMesh files.

    Parameters
    ----------
    filename : str
        Filename
    max_length : _type_, optional
        Maximum header length, by default MAX_HEADER_LENGTH. An exception will be raised if header
        is longer than this. It includes comments and blank lines.

    Returns
    -------
    Tuple[int, int, dict]
        A 3-tuple consisting of (count, offset, header) where count is the number of items to be
        read from the rest of the file, offset is the byte position in the file where count was
        read and header is the "FoamFile" dictionary.
    """

    count = 0
    header = {}

    with open(filename, "rb") as fh:
        n = 0
        while True:
            n += 1
            if n > max_length:
                raise RuntimeError(f"Exceeded maximum header length {max_length} during read")
            data = fh.readline()
            if not header:
                if data.strip() == b"FoamFile":
                    header = __read_dictionary(fh)
            if not count:
                if data.strip().isdigit():
                    count = int(data)
            if header and count:
                offset = fh.tell()
                break

    if "format" not in header:
        raise ValueError(f'"format" key not found in FoamFile dictionary while reading {filename}')

    if header["format"] not in ["binary", "ascii"]:
        raise ValueError(f'Unknown format {header["format"]} in FoamFile dictionary while reading {filename}')

    return count, offset, header


def __read_dictionary(file_handle: Union[BinaryIO, TextIO], max_length=20) -> dict:
    """Helper routine to parse the "FoamFile" dictionary in the header of OpenFOAM files.

    Parameters
    ----------
    file_handle : Union[BinaryIO, TextIO]
        File handle
    max_length : _type_, optional
        Maximum dictionary length, by default MAX_DICT_LENGTH. An exception will be raised if
        exceeded.

    Returns
    -------
    dict
        "FoamFile" dictionary
    """

    result = {}
    data = file_handle.readline().decode(encoding="utf-8")
    if not data.startswith("{"):
        raise ValueError
    for _ in range(max_length):
        data = file_handle.readline().decode(encoding="utf-8").replace(";", "").split()
        # Handle empty lines
        if not data:
            continue
        if data[0].startswith("}"):
            return result
        k = data[0].strip()
        v = " ".join(data[1:]).strip()
        for fun in [str]:
            try:
                result[k] = fun(v)
                break
            except ValueError:
                pass
    raise RuntimeError(f'Exceeded maximum dictionary length {max_length} during read while looking for "}}"')


def __remove_comments(text: str) -> str:
    """Remove C/C++ style comments from a given string (https://stackoverflow.com/a/241506)."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "
        else:
            return s

    pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)
    return re.sub(pattern, replacer, text)


def __remove_empty_lines(text: str, strip_lines=True, linesep=None) -> str:
    """Remove empty lines from a given string."""

    from os import linesep as _linesep

    if linesep is None:
        linesep = _linesep

    if strip_lines:
        return linesep.join([line.strip() for line in text.splitlines() if line.strip()])
    else:
        return linesep.join([line for line in text.splitlines() if line.strip()])


def __protected_split(text: str, sep=" \t", protected=("(", ")")) -> List[str]:
    """Split string while protecting contents within given container.

    The seperator can be multiple characters. Taken from https://stackoverflow.com/a/42070578
    """

    text = text.strip(sep)

    if (text == protected[0]) or (text == protected[1]):
        return [text]

    indices = [0]

    remainder = 0
    for i, c in enumerate(text):
        if c == protected[0]:
            remainder += 1
        elif c == protected[1]:
            remainder -= 1
        elif c in sep and remainder == 0:
            indices.append(i)
        if remainder < 0:
            raise SyntaxError(f"Missing {protected[0]}")

    indices.append(len(text))

    if remainder > 0:
        raise SyntaxError(f"Missing {protected[1]}")

    return [word for word in [text[i:j].strip(sep) for i, j in zip(indices, indices[1:])] if word]


def __dictionary_to_string(name: str, dictionary: dict) -> str:
    """Convert a single-level OpenFOAM dictionary to a string."""

    result = f"{name}\n{{"
    for k, v in dictionary.items():
        result += f"\n\t{k} {str(v)};"
    result += "\n}"

    return result


def read_faces(filename: str, dtype: DTypeLike, byteorder="little", verbose=1) -> Tuple[NDArray, NDArray]:
    """Reads the "faces" file and returns face definitions.

    The format of the file is taken from the "format" entry in the FoamFile dict.

    Parameters
    ----------
    filename : str
        Filename
    dtype : DTypeLike
        The corresponding numpy dtype, use ``np.int32`` for 32-bit integers and ``np.int64`` for
        64-bit integers.
    byteorder : str, optional
        Byte order while reading binary files, by default "little".
    verbose : int, optional
        Level of verbosity, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        A 2-tuple containing face point index and face point arrays.

    Note
    ----
    The face point index and face point arrays are defined as follows:

    Consider 3 faces formed by the following set of 3 ordered point sequences
    ``[0, 1, 7], [2, 1, 7, 6], [3, 2, 6, 5, 4]``. For this case, the face point index and face point
    arrays would be given as,

    >>> # face_point_index = array([0, 3, 7, 12])  # cumulative sum of point counts
    >>> # face_point_list = array([0, 1, 7, 2, 1, 7, 6, 3, 2, 6, 5, 4])

    Such that definition of "face 1" can be retrieved as,

    >>> face_point_list[face_point_index[1] : face_point_index[2]]  # get definition of face 1
    array([2, 1, 7, 6])
    """

    if verbose > 0:
        tic = time.perf_counter()

    count, offset, header = __parse_header(filename)

    item_size = np.dtype(dtype).itemsize

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        # Ensure the given dtype has the specified byteorder
        byteorder = "<" if byteorder == "little" else ">"
        dtype = np.dtype(dtype).newbyteorder(byteorder)
        with open(filename, "rb") as fh:
            if verbose > 0:
                print(f"Reading binary file {filename} ...", end=" ")
            fh.seek(offset + 1)
            buf = fh.read(item_size * count)
            indices = np.frombuffer(buf, dtype=dtype)
            fh.seek(1, 1)
            count = int(fh.readline())
            fh.seek(1, 1)
            buf = fh.read(item_size * count)
            points = np.frombuffer(buf, dtype=dtype)
    elif header["format"] == "ascii":
        _indices = [0]
        _points = []
        with open(filename, "r") as fh:
            if verbose > 0:
                print(f"Reading ascii file {filename} ...", end=" ")
            fh.seek(offset)
            # Read all lines, pre-split at newline characters
            lines = fh.readlines()
        for line in lines[1:-1]:
            _indices.append(line[0])
            _points.extend([point for point in line[2:-2].split()])
        indices = np.cumsum(np.array(_indices, dtype=dtype), dtype=dtype)
        points = np.array(_points, dtype=dtype)

    if verbose > 0:
        toc = time.perf_counter()
        print(f"read {len(indices) -1 } faces in {toc - tic:0.4f} seconds.", flush=True)

    return indices, points


def write_faces(
    filename: str,
    point_indices: NDArray,
    point_list: NDArray,
    header: dict,
    byteorder="little",
    encoding="utf-8",
    verbose=1,
) -> None:
    """Writes the faces file.

    Parameters
    ----------
    filename : str
        Filename
    point_indices : NDArray
        Face point index array
    point_list : NDArray
        Face point list array
    header : dict
        "FoamFile" dictionary to be placed in the header
    dtype : DTypeLike
        The numpy dtype to use for binary output, by default np.int32
    byteorder : str, optional
        Byte order for binary writing, by default "little"
    encoding : str, optional
        Encoding for ascii writing, by default "utf-8"
    verbose : int, optional
        Level of verbosity, by default 1
    """

    if verbose > 0:
        tic = time.perf_counter()

    if point_indices.ndim != 1:
        raise ValueError(
            f"Face point index array should be one dimensional, but it has {point_indices.ndim} dimensions"
        )
    if point_list.ndim != 1:
        raise ValueError(f"Face point list array should be one dimensional, but it has {point_indices.ndim} dimensions")

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        if point_indices.dtype.byteorder != point_list.dtype.byteorder:
            raise RuntimeError("Different byte orders in face point index and face point list arrays")
        # Ensure the given dtype has the specified byteorder
        current = point_indices.dtype.byteorder
        if current == "=":
            current = sys.byteorder
        elif current == "<":
            current = "little"
        elif current == ">":
            current = "big"
        else:
            raise RuntimeError
        swap = False
        if current == "little" and byteorder == "big":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        elif current == "big" and byteorder == "little":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        # Write
        with open(filename, "wb") as fh:
            if verbose > 0:
                print(f"Writing binary file {filename} ...", end=" ")
            fh.write(bytes(__dictionary_to_string("FoamFile", header), encoding))
            fh.write(bytes(f"\n\n{point_indices.size}\n(", encoding))
            (fh.write(point_indices.byteswap().tobytes()) if swap else fh.write(point_indices.tobytes()))
            fh.write(bytes(f"){point_list.size}\n", encoding))
            fh.write(b"(")
            (fh.write(point_list.byteswap().tobytes()) if swap else fh.write(point_list.tobytes()))
            fh.write(b")")
    elif header["format"] == "ascii":
        n_faces = point_indices.size - 1
        with open(filename, "w") as fh:
            if verbose > 0:
                print(f"Writing ascii file {filename} ...", end=" ")
            fh.write(__dictionary_to_string("FoamFile", header) + f"\n\n{n_faces}\n(\n")
            for face in range(n_faces):
                face_points = point_list[point_indices[face] : point_indices[face + 1]]
                fh.write(f"{len(face_points)}({' '.join([str(point) for point in face_points])})\n")
            fh.write(")")
    else:
        raise ValueError(f'Unknown format {header["format"]} specified in FoamFile dictionary')

    if verbose > 0:
        toc = time.perf_counter()
        print(f"wrote {point_indices.size - 1} faces in {toc - tic:0.4f} seconds.", flush=True)


def read_vector_field(filename: str, dtype: DTypeLike, ndims=3, byteorder="little", verbose=1) -> NDArray:
    """Reads a vector field.

    The format of the file is taken from the "format" entry in the FoamFile dict.

    Parameters
    ----------
    filename : str
        Filename
    dtype : DTypeLike
        The corresponding numpy dtype, e.g. ``np.float64``
    ndims : int, optional
        Number of spatial dimensions, by default 3
    byteorder : str, optional
        Byte order for binary reading, by default "little"
    verbose : int, optional
        Level of verbosity, by default 1

    Returns
    -------
    NDArray
        Vector field, shape(:, ndims)
    """

    if verbose > 0:
        tic = time.perf_counter()

    count, offset, header = __parse_header(filename)

    item_size = np.dtype(dtype).itemsize

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        # Ensure the given dtype has the specified byteorder
        byteorder = "<" if byteorder == "little" else ">"
        dtype = np.dtype(dtype).newbyteorder(byteorder)
        with open(filename, "rb") as fh:
            if verbose > 0:
                print(f"Reading binary file {filename} ...", end=" ")
            fh.seek(offset + 1)
            buf = fh.read(item_size * count * ndims)
        data = np.frombuffer(buf, dtype=dtype).reshape(count, ndims)
    elif header["format"] == "ascii":
        with open(filename, "r") as fh:
            if verbose > 0:
                print(f"Reading ascii file {filename} ...", end=" ")
            fh.seek(offset)
            # Read all lines and split at newline characters
            lines = fh.readlines()
        data = np.array([line[1:-2].split() for line in lines[1 : count + 1]], dtype=dtype)

    if verbose > 0:
        toc = time.perf_counter()
        print(
            f"read {count} vectors ({str(np.dtype(dtype))}) in {toc - tic:0.4f} seconds.",
            flush=True,
        )

    # NOTE: For some large files, the array "data" was set to read-only and not modifiable,
    # which seems to be a bug with numba 0.58.1. Below is a workaround to ensure a writeable array
    # is returned.
    return np.array(data)


def write_vector_field(
    filename: str,
    data: NDArray,
    header: dict,
    byteorder="little",
    encoding="utf-8",
    format="%.18f",
    verbose=1,
) -> None:
    """Writes a vector field.

    Parameters
    ----------
    filename : str
        Filename
    data : NDArray
        Vector field to write, shape(:, ndims)
    header : dict
        "FoamFile" dictionary to be placed in the header
    byteorder : str, optional
        Byte order for binary writing, by default "little"
    encoding : str, optional
        Encoding for ascii writing, by default "utf-8"
    format : str, optional
        Number format for ascii writing, by default "%.18f"
    verbose : int, optional
        Level of verbosity, by default 1
    """

    if verbose > 0:
        tic = time.perf_counter()

    if data.ndim != 2:
        raise ValueError(f"Input array should be two dimensional, but it has {data.ndim} dimensions")

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        # Ensure the given dtype has the specified byteorder
        current = data.dtype.byteorder
        if current == "=":
            current = sys.byteorder
        elif current == "<":
            current = "little"
        elif current == ">":
            current = "big"
        else:
            raise RuntimeError
        swap = False
        if current == "little" and byteorder == "big":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        elif current == "big" and byteorder == "little":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        # Write
        with open(filename, "wb") as fh:
            if verbose > 0:
                print(f"Writing binary file {filename} ...", end=" ")
            fh.write(bytes(__dictionary_to_string("FoamFile", header), encoding))
            fh.write(bytes(f"\n\n{data.shape[0]}\n(", encoding))
            (fh.write(data.flatten().byteswap().tobytes()) if swap else fh.write(data.flatten().tobytes()))
            fh.write(b")")
    elif header["format"] == "ascii":
        format = "(" + " ".join(data.shape[1] * [format]) + ")"
        if verbose > 0:
            print(f"Writing ascii file {filename} ...", end=" ")
        np.savetxt(
            filename,
            data,
            fmt=format,
            header=__dictionary_to_string("FoamFile", header) + f"\n\n{data.shape[0]}\n(",
            footer=")",
            comments="",
        )
    else:
        raise ValueError(f'Unknown format {header["format"]} specified in FoamFile dictionary')

    if verbose > 0:
        toc = time.perf_counter()
        print(
            f"wrote {data.shape[0]} vectors ({str(data.dtype)}) in {toc - tic:0.4f} seconds.",
            flush=True,
        )


def read_scalar_field(filename: str, dtype: DTypeLike, byteorder="little", verbose=1) -> NDArray:
    """Reads a scalar field.

    Parameters
    ----------
    filename : str
        Filename
    dtype : DTypeLike
        The corresponding numpy dtype, e.g. ``np.float64`` or ``np.int32`` etc.
    byteorder : str, optional
        Byte order for binary reading, by default "little"
    verbose : int, optional
        Level of verbosity, by default 1

    Returns
    -------
    NDArray
        Scalar field, shape(:)
    """

    if verbose > 0:
        tic = time.perf_counter()

    count, offset, header = __parse_header(filename)

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        # Ensure the given dtype has the specified byteorder
        byteorder = "<" if byteorder == "little" else ">"
        dtype = np.dtype(dtype).newbyteorder(byteorder)
        with open(filename, "rb") as fh:
            if verbose > 0:
                print(f"Reading binary file {filename} ...", end=" ")
            fh.seek(offset + 1)
            buf = fh.read(np.dtype(dtype).itemsize * count)
        data = np.frombuffer(buf, dtype=dtype)
    elif header["format"] == "ascii":
        with open(filename, "r") as fh:
            if verbose > 0:
                print(f"Reading ascii file {filename} ...", end=" ")
            fh.seek(offset)
            # Read all lines, pre-split at newline characters
            lines = fh.readlines()
        # data = np.array(lines[1:-1], dtype=dtype)
        data = np.array(lines[1 : count + 1], dtype=dtype)

    if verbose > 0:
        toc = time.perf_counter()
        print(
            f"read {count} scalars ({str(np.dtype(dtype))}) in {toc - tic:0.4f} seconds.",
            flush=True,
        )

    return np.array(data)


def write_scalar_field(
    filename: str,
    data: NDArray,
    header: dict,
    byteorder="little",
    encoding="utf-8",
    format="%d",
    verbose=1,
) -> None:
    """Writes a scalar field.

    Note that the default value of "format" is adjusted to an integer field and require sadjustment
    for float fields.

    Parameters
    ----------
    filename : str
        Filename
    data : NDArray
        Scalar field to write, shape(:)
    header : dict
        "FoamFile" dictionary to be placed in the header
    byteorder : str, optional
        Byte order for binary writing, by default "little"
    encoding : str, optional
        Encoding for ascii writing, by default "utf-8"
    format : str, optional
        Number format for ascii writing, by default "%d"
    verbose : int, optional
        Level of verbosity, by default 1
    """

    if verbose > 0:
        tic = time.perf_counter()

    if data.ndim != 1:
        raise ValueError(f"Input array should be one dimensional, but it has {data.ndim} dimensions")

    if header["format"] == "binary":
        if byteorder not in ["big", "little"]:
            raise ValueError(
                f'Invalid value "{byteorder}" for "byteorder", valid options are "little" (default), and "big"'
            )
        # Ensure the given dtype has the specified byteorder
        current = data.dtype.byteorder
        if current == "=":
            current = sys.byteorder
        elif current == "<":
            current = "little"
        elif current == ">":
            current = "big"
        else:
            raise RuntimeError
        swap = False
        if current == "little" and byteorder == "big":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        elif current == "big" and byteorder == "little":
            swap = True
            if verbose > 0:
                print(f'Swapping byte order from "{current}" to "{byteorder}"')
        # Write
        with open(filename, "wb") as fh:
            if verbose > 0:
                print(f"Writing binary file {filename} ...", end=" ")
            fh.write(bytes(__dictionary_to_string("FoamFile", header), encoding))
            fh.write(bytes(f"\n\n{data.size}\n(", encoding))
            fh.write(data.byteswap().tobytes()) if swap else fh.write(data.tobytes())
            fh.write(b")")
    elif header["format"] == "ascii":
        if verbose > 0:
            print(f"Writing ascii file {filename} ...", end=" ")
        np.savetxt(
            filename,
            data,
            fmt=format,
            header=__dictionary_to_string("FoamFile", header) + f"\n\n{data.size}\n(",
            footer=")",
            comments="",
        )
    else:
        raise ValueError(f'Unknown format {header["format"]} specified in FoamFile dictionary')

    if verbose > 0:
        toc = time.perf_counter()
        print(
            f"wrote {data.size} scalars ({str(data.dtype)}) in {toc - tic:0.4f} seconds.",
            flush=True,
        )


def recursive_dictionary_parser(filename: str, **kwargs) -> Union[dict, int]:
    """Read OpenFOAM dictionary files.

    Instead of creating a Parser class, this routine offers similar functionality using recursive
    calls. The read dictionaries can be written again with ``recursive_dictionary_writer``.

    Parameters
    ----------
    filename : str
        Filename

    Keyword arguments
    -----------------
    verbose : int
        Level of verbosity, by default 0.
    parent : str
        Name for unnamed dictionaries at root level, by default base filename is used.

    Returns
    -------
    Union[dict, int]
        Parsed dictionary. Integer values are returned during recursive calls.

    Note
    ----
    In theory this function should be able to read all dictionary file types, with arbitrary level
    of depth. However, it is tested with a limited number of examples and hence may not behave
    as expected for all cases.
    """

    # Check if this was a recursive call or user call
    user_call = kwargs.get("user_call", True)

    # Level of verbosity
    verbose = kwargs.get("verbose", 0)

    # Handle user call
    if user_call:
        # Read file
        if verbose > 0:
            print(f"Reading {filename} ...", end=" ")
        with open(filename, "r") as fh:
            lines = fh.read()
        lines = __remove_empty_lines(__remove_comments(lines)).splitlines()
        if verbose > 0:
            print(
                f"processing {len(lines)} lines after removing comments and blank lines ...",
                flush=True,
            )
        # Initialize container and cursor
        cursor = 0  # current position (line) in file
        container = OrderedDict()
        # Use the base filename as the default initial parent key for unnamed sequences
        parent = kwargs.get("parent", os.path.splitext(os.path.basename(filename))[0])
        # Log recursion depth for debug information
        depth = 0
    # Handle self (recursive) call
    else:
        lines = kwargs.get("lines")
        cursor = kwargs.get("cursor")
        container = kwargs.get("container")
        parent = kwargs.get("parent")
        depth = kwargs.get("depth")

    if verbose > 1:
        print(f">>> ({depth}) cursor={cursor}, parent={parent}, container={str(type(container))}")

    while True:

        # Process until EOF is reached
        if cursor > len(lines) - 1:
            if verbose > 2:
                print("    break: reached end-of-file")
            # EOF should only be reached at depth 0, otherwise there is an error
            if depth != 0:
                raise ValueError("Reached EOF while searching for a section termination")
            break

        line = lines[cursor]

        try:
            next_line = lines[cursor + 1]
        except IndexError:
            next_line = ""

        try:
            next_next_line = lines[cursor + 2]
        except IndexError:
            next_next_line = ""

        if verbose > 2:
            print(
                f'    ({depth}) cursor={cursor}, line="{line}", next_line="{next_line}",'
                f' next_next_line="{next_next_line}"'
            )

        # Increment the cursor prematurely to avoid incrementing at each time we break out of loop
        cursor += 1

        # Break the line into words, but do not split within paranthesis, e.g. (word1 word2)
        words = [word for word in __protected_split(line.replace(";", ""))]

        # End of statements with semicolon, e.g. dictionary entries, single-word list entries
        if line.endswith(";"):
            # No whitespace
            if len(words) == 1:
                # End of list
                if words[0] == ")":
                    if verbose > 2:
                        print(f"!!! ({depth}) break: end of list with semicolon")
                    container.info.__setitem__("end", ");")
                    break
                # End of dictionary of dictionaries
                elif words[0] == "}":
                    if verbose > 2:
                        print(f"!!! ({depth}) break: end of dictionary with semicolon")
                    container.info.__setitem__("end", "};")
                    break
                else:
                    raise ValueError(
                        "Attempted to read single word not in [), }}] terminated with a semicolon,"
                        " missing dictionary value?"
                    )
            # Dictionary key and value pair
            else:
                k = words[0]
                v = words[1:]
                # Single value
                if len(v) == 1:
                    if verbose > 2:
                        print(f'    ({depth}) reading key and single-valued value pair, k="{k}", v={v}')
                    # Try to cast into relevant form
                    for fun in [int, float, str]:
                        try:
                            container[k] = fun(v[0])
                            if verbose > 3:
                                print(f'    ({depth}) casted value "{v[0]}" to {str(fun)}')
                            break
                        except ValueError:
                            pass
                # Multiple values
                else:
                    mv = [word for word in __protected_split(" ".join(words[1:]), protected=("[", "]"))]
                    container[k] = mv
                    # raise NotImplementedError(
                    #     "Reading key value pairs with multiple values is not implemented"
                    # )
        # A section beginning
        else:
            if len(words) == 1:
                # End of a dictionary
                if line == "}":
                    if verbose > 2:
                        print(f"!!! ({depth}) break: end of dictionary without semicolon")
                    container.info.__setitem__("end", "}")
                    break
                # End of a list without semicolon, only possible at end-of-file
                if line == ")":
                    if verbose > 1:
                        print(f"!!! ({depth}) break: end of list without semicolon")
                    if cursor != len(lines):  # not len(lines) - 1 because we have already incremented the cursor
                        raise ValueError("Ending a list without a semicolon is only possible at end-of-file")
                    container.info.__setitem__("end", ")")
                    break
                # Start of an unnamed dictionary
                if line == "{":
                    if verbose > 2:
                        print(
                            f"+++ ({depth}) starting recursive call to read unnamed dictionary, using parent={parent}"
                        )
                    container[len(container)] = OrderedDict()
                    container[len(container) - 1].info = {}
                    container[len(container) - 1].info.__setitem__("type", "unnamed_dict")
                    cursor = recursive_dictionary_parser(
                        filename,
                        lines=lines,
                        cursor=cursor,
                        container=container[len(container) - 1],
                        parent=str(len(container) - 1),
                        user_call=False,
                        verbose=verbose,
                        depth=depth + 1,
                    )
                    if verbose > 2:
                        print(
                            f"--- ({depth}) recursive call reading unnamed dictionary returns with"
                            f" cursor={cursor}, starting next line"
                        )
                    continue
                if line == "(":
                    raise RuntimeError('Read a line containing only "(", this line should have already been skipped!')
                # Start of an unnamed list
                if line.isnumeric():
                    if next_line == "(":
                        # This denotes count of items
                        item_count = int(line)
                        if verbose > 2:
                            print(
                                f"+++ ({depth}) starting recursive call to read list with an item"
                                f" count of {item_count}, using parent={parent}"
                            )
                        container[parent] = OrderedDict()
                        container[parent].info = {}
                        container[parent].info.__setitem__("type", "counted_unnamed_list")
                        # Recursive call
                        cursor = recursive_dictionary_parser(
                            filename,
                            lines=lines,
                            cursor=cursor + 1,
                            container=container[parent],
                            parent=parent,
                            user_call=False,
                            verbose=verbose,
                            depth=depth + 1,
                        )
                        if verbose > 2:
                            print(
                                f"--- ({depth}) recursive call reading a list returns with"
                                f" cursor={cursor}, starting next line"
                            )
                        # Check element count
                        if len(container[parent]) != item_count:
                            raise ValueError(f"Expected {item_count} items in list, found {len(container[parent])}")
                        continue
                    else:
                        raise ValueError(f'Expected start of list "(" after an integer line, found {line}')
                # Start of a named container
                else:
                    if any(char.isalpha() for char in line):
                        # Dictionary
                        if next_line == "{":
                            dict_name = line
                            if verbose > 2:
                                print(
                                    f"+++ ({depth}) starting recursive call to dictionary named"
                                    f' "{dict_name}", using parent={parent}'
                                )
                            container[dict_name] = OrderedDict()
                            container[dict_name].info = {}
                            container[dict_name].info.__setitem__("type", "named_dict")
                            # For recursive calls inferred from the next line, cursor is
                            # incremented to avoid reading e.g. "(" and "{"
                            cursor = recursive_dictionary_parser(
                                filename,
                                lines=lines,
                                cursor=cursor + 1,
                                container=container[dict_name],
                                parent=dict_name,
                                user_call=False,
                                verbose=verbose,
                                depth=depth + 1,
                            )
                            if verbose > 2:
                                print(
                                    f'--- ({depth}) recursive call reading dictionary "{dict_name}"'
                                    f" returns with cursor={cursor}, starting next line"
                                )
                            continue
                        # Named list with unspecified number of items
                        elif next_line == "(":
                            list_name = line
                            if verbose > 2:
                                print(
                                    f"+++ ({depth}) starting recursive call to read named list"
                                    f' "{list_name}", using parent={parent}'
                                )
                            container[list_name] = OrderedDict()
                            container[list_name].info = {}
                            container[list_name].info.__setitem__("type", "named_list")
                            cursor = recursive_dictionary_parser(
                                filename,
                                lines=lines,
                                cursor=cursor + 1,
                                container=container[list_name],
                                parent=list_name,
                                user_call=False,
                                verbose=verbose,
                                depth=depth + 1,
                            )
                            if verbose > 2:
                                print(
                                    f'--- ({depth}) recursive call reading list "{list_name}"'
                                    f" returns with cursor={cursor}, starting next line"
                                )
                            continue
                        # List with class name identifier
                        elif next_line.startswith("List"):
                            raise NotImplementedError("Lists with class name identifier is not implemented")
                        # Named list with specified number of entries
                        elif next_line.isnumeric():
                            raise NotImplementedError("Named list with specified number of entires is not implemented")
                        # A single-word list entry
                        else:
                            if "list" not in container["__type__"]:
                                raise RuntimeError("Unhandled case or missing section termination")
                            # Try to cast into relevant form
                            for fun in [int, float, str]:
                                try:
                                    container[fun(words[0])] = None
                                    if verbose > 3:
                                        print(f'    ({depth}) casted value "{words[0]}" to {str(fun)}')
                                    break
                                except ValueError:
                                    pass
                    else:
                        raise RuntimeError("Unhandled case: line is not an integer and does not contain any letters")
            else:
                raise RuntimeError("Unhandled case: multiple space-seperated content without semicolon ending")

    if verbose > 2:
        if user_call:
            print(f"<<< ({depth}) user call returns", end="")
        else:
            print(f"<<< ({depth}) cursor={cursor}, line={line}")

    return container if user_call else cursor


def recursive_dictionary_writer(filename: str, container: Mapping, **kwargs) -> None:
    """Write OpenFOAM dictionary files.

    Can directly write content read by ``recursive_dictionary_parser``. Refer to the
    "container definition" before attempting to write user-generated dictionaries.

    For this

    Parameters
    ----------
    filename : str
        Filename
    container : Mapping
        Container to write

    Container definition
    --------------------
    Container is basically a dictionary which can have arbitrary number of sub-dictionaries, e.g.
    arbitrary depth of nestedness. Only requirement is that, the ``container`` itself and nested
    containers within should contain an "info" attribute. The info attribute should be a dictionary
    as ``container.info = {"end": str, "type": str}``.

    The key ``end`` specifies the character to use while ending this container, common options are
    ")", ");", "}" and "};".

    The key ``type`` specifies the type of container, and should be one of the following:

    ``unnamed_dict`` : Unnamed dictionary

    ``named_dict`` : Named dictionary, e.g. as in boundary definitons

    ``counted_unnamed_list`` : Unnamed list with given element count, e.g. as in boundary list

    ``named_list`` : Named list

    Keyword arguments
    -----------------
    tab_char : str
        Character to use for tabs, by default "\\t"
    verbose : int
        Level of verbosity, by defalt 0
    """

    tab_char = kwargs.get("tab_char", "\t")

    # Check if this was a recursive call or user call
    user_call = kwargs.get("user_call", True)

    # Level of verbosity
    verbose = kwargs.get("verbose", 0)

    if not isinstance(container, dict):
        raise TypeError("Parameter container has to be a dictionary")

    # Handle user call
    if user_call:
        if verbose > 0:
            print(f"Writing {filename} ...")
        # Create file handle
        file_handle = open(filename, "w")
        if verbose > 1:
            print("*** opened file")
        # Log recursion depth for debug information
        depth = 0
    # Handle self (recursive) call
    else:
        depth = kwargs.get("depth")
        file_handle = kwargs.get("file_handle")

    if verbose > 1:
        print(f">>> ({depth}) position={file_handle.tell()}")

    for k, v in container.items():
        if verbose > 2:
            print(f"    ({depth}) k={k}, type(v)={str(type(v))}")
        if isinstance(v, dict):
            if not hasattr(v, "info"):
                raise RuntimeError('Missing attribute "info" in container')
            if v.info["type"] == "unnamed_dict":
                file_handle.write(f"{depth*tab_char}{{\n")
                recursive_dictionary_writer(
                    filename,
                    v,
                    depth=depth + 1,
                    file_handle=file_handle,
                    user_call=False,
                    verbose=verbose,
                )
            elif v.info["type"] == "named_dict":
                file_handle.write(f"{depth*tab_char}{str(k)}\n")
                file_handle.write(f"{depth*tab_char}{{\n")
                if verbose > 3:
                    print("+++ ")
                recursive_dictionary_writer(
                    filename,
                    v,
                    depth=depth + 1,
                    file_handle=file_handle,
                    user_call=False,
                    verbose=verbose,
                )
                if verbose > 3:
                    print("--- ")
            elif v.info["type"] == "counted_unnamed_list":
                file_handle.write(f"{depth*tab_char}{len(v)}\n")
                file_handle.write(f"{depth*tab_char}(\n")
                recursive_dictionary_writer(
                    filename,
                    v,
                    depth=depth + 1,
                    file_handle=file_handle,
                    user_call=False,
                    verbose=verbose,
                )
            elif v.info["type"] == "named_list":
                file_handle.write(f"{depth*tab_char}{str(k)}\n")
                file_handle.write(f"{depth*tab_char}(\n")
                recursive_dictionary_writer(
                    filename,
                    v,
                    depth=depth + 1,
                    file_handle=file_handle,
                    user_call=False,
                    verbose=verbose,
                )
            else:
                raise RuntimeError
            file_handle.write(f'{depth*tab_char}{v.info["end"]}\n')
            continue
        elif isinstance(v, str):
            if verbose > 4:
                print(f"    ({depth}) writing {k}={v}")
            file_handle.write(f"{depth*tab_char}{str(k)}{tab_char}{v};\n")
            continue
        elif isinstance(v, int):
            file_handle.write(f"{depth*tab_char}{str(k)}{tab_char}{v};\n")
            continue
        elif isinstance(v, float):
            file_handle.write(f"{depth*tab_char}{str(k)}{tab_char}{v};\n")
            continue
        elif v is None:
            file_handle.write(f"{depth*tab_char}{str(k)}\n")
            continue
        else:
            raise ValueError(f'Encountered unsupported type "{str(type(v))} in key {k}')

    if verbose > 1:
        print(f"<<< ({depth}), position={file_handle.tell()}")

    if user_call:
        file_handle.close()
        if verbose > 1:
            print("*** closed file")

    return None
