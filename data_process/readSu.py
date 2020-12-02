from obspy import read
import numpy as np

def read_su_trace(filePath):
    data = read(filePath, unpack_trace_headers=True)
    data_all = []
    for tr in data.traces:
        data_all.append(tr.data)
    return np.array(data_all)

def read_su(filePath):
    data = read(filePath, unpack_trace_headers=True)
    ntr = len(data.traces)
    ns = len(data.traces[0].data)
    dt = data.traces[0].stats.delta

    # x- and y-source coordinates from trace header
    xsrc = data.traces[0].stats.su.trace_header.source_coordinate_x
    ysrc = data.traces[0].stats.su.trace_header.source_coordinate_y

    # allocate memory for traces and receiver positions
    traces = np.zeros((ns, ntr))
    xrec = np.zeros((1, ntr))
    yrec = np.zeros((1, ntr))

    i = 0
    for tr in data.traces:
        # extract traces
        traces[:, i] = tr.data[:]

        # x- and y-receiver coordinates from trace header
        xrec[0, i] = data.traces[i].stats.su.trace_header.group_coordinate_x
        yrec[0, i] = data.traces[i].stats.su.trace_header.group_coordinate_y

        i += 1

    # flip traces
    traces = np.flipud(traces)

    # offset [km]
    offset = (xrec - xsrc) / 1e6
    return traces
