import time
import argparse
import random
from typing import Tuple
from torch import Tensor, nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
scaler = StandardScaler()
# print(torch.cuda.is_available())
# from torch_scatter import scatter

"""# Preprocessing"""

class Core():
    """
    A class used to represent a Core in a benchmarking system.
    Attributes
    ----------
    id : int
        The unique identifier for the core.
    benchmark : str
        The benchmark associated with the core.
    core_no : int
        The core index.
    patterns : int
        Number of patterns associated with the core.
    scan : int
        Scan chain length for the core
    Methods
    -------
    __init__(self, id, benchmark, core_no, patterns, scan)
        Initializes the Core with the given id, benchmark, core number, patterns, and scanchain length
    """

    def __init__(self, id, benchmark, core_no, patterns, scan):
        self.id = id
        self.benchmark = benchmark
        self.core_no = core_no
        self.patterns = patterns
        self.scan = scan

class IOPair():
    """
    IOPair class represents a pair of input (src) and output (sink).

    Attributes:
        src: The source core index.
        sink: The sink core index.

    Methods:
        __init__(self, src, sink): Initializes the IOPair with a source and a sink.
    """
    def __init__(self, src, sink):
        self.src = src
        self.sink = sink


def generateSamples(num_samples):
    """
    Generates a specified number of samples, each containing a 64x2 array of directional data.

    Args:
        num_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: A tensor containing the generated samples.
    """
    data = []
    for i in range(num_samples):
        dir = np.zeros(shape=[65,2])
        for i in range(0,65):
            if i%4==1:
                if i<32:
                    dir[i,1]=0
                    dir[i,0]=i//4
                else:
                    dir[i,1]=4
                    dir[i,0]=(i-32)//4
            elif i%4==2:
                if i<32:
                    dir[i,1]=1
                    dir[i,0]=i//4
                else:
                    dir[i,1]=5
                    dir[i,0]=(i-32)//4
            elif i%4==3:
                if i<32:
                    dir[i,1]=2
                    dir[i,0]=i//4
                else:
                    dir[i,1]=6
                    dir[i,0]=(i-32)//4
            else:
                if i<=32:
                    dir[i,1]=3
                    dir[i,0]=i//4 - 1
                else:
                    dir[i,1]=7
                    dir[i,0]=(i-32)//4 - 1
        dir = dir[1:]
        data.append(dir)
    data = torch.from_numpy(np.array(data).astype('float32'))
    return data

def visualize_graph(graph):
    # Draw the graph using NetworkX and Matplotlib
    pos = nx.get_node_attributes(graph, 'pos')
    labels = {node: data['label'] for node, data in graph.nodes(data=True) if data['label'] is not None}
    nx.draw(graph, pos, labels=labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Display the graph
    plt.title("Graph Visualization")
    plt.show()


def create_data(batch_size, num_cores, cols):
    """
    Generates a batch of data representing core positions in a grid.

    Args:
        batch_size (int): The number of data samples to generate.
        num_cores (int): The total number of cores.
        cols (int): The number of columns in the grid.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_cores, 2) containing the positions of the cores.
    """
    core_positions = []

     # Calculate the row and column for each core
    for i in range(num_cores):
        row = (i // cols) + 1  # Convert to 1-based index, used only to draw the NoC
        col = (i %  cols) + 1  # Convert to 1-based index, used only to draw the NoC
        core_positions.append((row, col))

    data=[]
    dir = np.zeros(shape=[num_cores,2])
    for i in range(num_cores):
        dir[i,0] = (i // cols) # 0 based index, dir[i,0] -> x coordinate of the core
        dir[i,1] = (i %  cols) # 0 based index, dir[i,1] -> y coordinate of the core
    for i in range(batch_size):
        data.append(dir)
    data = torch.from_numpy(np.array(data).astype('float32'))

    return core_positions, data


def generate_positions(mesh_dims):
    """Generate core positions (1 based) for a given mesh dimension."""
    if mesh_dims is None:
        raise ValueError("Mesh dimensions must be provided")
    rows, cols = mesh_dims
    return [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)]


def generate_edges(mesh_dims):
    """Generate edges for a grid with given dimensions."""
    edges = []
    rows, cols = mesh_dims
    # Add edges for rows
    for r in range(rows):
        for c in range(cols - 1):
            node = r * cols + c + 1
            edges.append((node, node + 1))

    # Add edges for columns
    for c in range(cols):
        for r in range(rows - 1):
            node = r * cols + c + 1
            edges.append((node, node + cols))

    return edges


def prep_data(num_cores, test=True):
    """
    Prepares data for a given number of cores.
    Parameters:
    num_cores (int): The number of cores to prepare data for.
    test (bool): If True, use benchmark data files. Default is True.
    Returns:
    tuple: A tuple containing:
        - data (various): The prepared data.
        - cores (list): A list of lists, each containing Core objects.
        - io (list): A list of lists, each containing I/O pairs.
        - dims (list): A list containing the dimensions of the graph.
        - graph (networkx.Graph): The graph representing the core connections.
    """

    file = open(f"data_{num_cores}cores.txt", 'r')
    if test:
        file = open(f"bm_{num_cores}cores.txt", 'r')

    lines = file.readlines()
    batch_size = len(lines)//num_cores
    cores = [[] for _ in range(batch_size)]
    for i in range(len(lines)):
        if i // num_cores >= batch_size: # extra precaution if number of lines in file is not an integer multiple of batch_size
            break
        line = lines[i]
        data = line.split()
        cores[i // num_cores].append((int(data[0]), data[1], int(data[2]), int(data[3]), int(data[4])))

    data = None
    graph = None
    dims = None

    if num_cores == 7:
        io = [[2, 7], [3, 6]]
        dims = [3, 3]
    elif num_cores == 8:
        io = [[1,2], [3,7]]
        dims = [2,4]
    elif num_cores == 10:
        io = [[1,7], [2,6], [3,10]]
        dims = [3,4]
    elif num_cores == 14:
        io = [[3,6], [4,10]]
        dims = [2,7]
    elif num_cores == 16:
        io = [[2,4], [5,9], [13,14], [12,16]]
        dims = [4,4]
    elif num_cores == 28:
        io = [[5,12], [8,16], [9,17], [13,20]]
        dims = [7,4]
    elif num_cores == 32:
        io = [[1,4], [9,12], [17,20], [25,28]]
        dims = [8,4]
    elif num_cores == 64:
        io = [[13,17], [33,40], [44,52], [61,63]]
        dims = [8,8]
    elif num_cores == 40:
        io = [[1, 35], [4,12], [30,32], [37, 39]]
        dims = [8,5]
    elif num_cores == 48:
        io = [[3,5], [21,30], [32,34], [38, 40]]
        dims = [8,6]
    elif num_cores == 56:
        io = [[3,5], [21,30], [32,34], [38, 40]]
        dims = [8,7]

    core_positions, data = create_data(batch_size, num_cores, dims[1])

    # Generate positions for a mxn mesh
    positions = generate_positions(dims)

    # # Add nodes for all positions, but only label core positions
    for idx, pos in enumerate(positions, start=1):
        label = idx if pos in core_positions else None
        print(idx, pos)
        graph.add_node(idx, pos=pos, label=label)

    # Define edges based on adjacency in the grid
    edges = generate_edges(dims)
    graph.add_edges_from(edges)

    return data, cores, io, dims, graph


def hop_counts(G, source, target):
    """
    Calculate the shortest path length (hop count) between two nodes in a graph.

    Parameters:
    G (networkx.Graph): The graph in which to find the shortest path.
    source (node): The starting node for the path.
    target (node): The ending node for the path.

    Returns:
    int: The number of edges in the shortest path from source to target.

    Raises:
    NetworkXNoPath: If there is no path between source and target.
    NodeNotFound: If either source or target is not in the graph.
    """
    return nx.shortest_path_length(G, source, target)

def check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2):
    """
    Checks if there is a path conflict between two sets of source, core, and sink coordinates.

    Args:
        dir (list): A list where elements are tuples of (y, x) coordinates.
        core1 (int): Identifier for the core node in the first path.
        src1 (int): Identifier for the source node in the first path.
        sink1 (int): Identifier for the sink node in the first path.
        core2 (int): Identifier for the core node in the second path.
        src2 (int): Identifier for the source node in the second path.
        sink2 (int): Identifier for the sink node in the second path.

    Returns:
        bool: True if there is a conflict between the two paths, False otherwise.
    """

    ys1 = dir[src1][0].item()
    xs1 = dir[src1][1].item()
    yc1 = dir[core1][0].item()
    xc1 = dir[core1][1].item()
    yk1 = dir[sink1][0].item()
    xk1 = dir[sink1][1].item()
    ys2 = dir[src2][0].item()
    xs2 = dir[src2][1].item()
    yc2 = dir[core2][0].item()
    xc2 = dir[core2][1].item()
    yk2 = dir[sink2][0].item()
    xk2 = dir[sink2][1].item()

  # print(xc1, yc1)
  # print(xs1, ys1)
  # print(xk1, yk1)
  # print(xc2, yc2)
  # print(xs2, ys2)
  # print(xk2, yk2)

    if (xs1-xc1)*(xs2-xc2) > 0:
        if not((xc1 <= min(xc2,xs2) and xs1 <= min(xc2,xs2)) or (xc1 >= max(xc2,xs2) and xs1 >= max(xc2,xs2))) and ys1==ys2:
            # print('F')
            return True
    if (xs1-xc1)*(xc2-xk2) > 0:
        if not((xc1 <= min(xc2,xk2) and xs1 <= min(xc2,xk2)) or (xc1 >= max(xc2,xk2) and xs1 >= max(xc2,xk2))) and ys1==yc2:
            # print('G')
            return True
    if (xc1-xk1)*(xs2-xc2) > 0:
        if not((xc1 <= min(xc2,xs2) and xk1 <= min(xc2,xs2)) or (xc1 >= max(xc2,xs2) and xk1 >= max(xc2,xs2))) and yc1==ys2:
            # print('H')
            return True
    if (xc1-xk1)*(xc2-xk2) > 0:
        if not((xc1 <= min(xc2,xk2) and xk1 <= min(xc2,xk2)) or (xc1 >= max(xc2,xk2) and xk1 >= max(xc2,xk2))) and yc1==yc2:
            # print('I')
            return True
    if (ys1-yc1)*(ys2-yc2) > 0:
        if not((yc1 <= min(yc2,ys2) and ys1 <= min(yc2,ys2)) or (yc1 >= max(yc2,ys2) and ys1 >= max(yc2,ys2))) and xc1==xc2:
            # print('J')
            return True
    if (ys1-yc1)*(yc2-yk2) > 0:
        if not((yc1 <= min(yc2,yk2) and ys1 <= min(yc2,yk2)) or (yc1 >= max(yc2,yk2) and ys1 >= max(yc2,yk2))) and xc1==xk2:
            # print('K')
            return True
    if (yc1-yk1)*(ys2-yc2) > 0:
        if not((yc1 <= min(yc2,ys2) and yk1 <= min(yc2,ys2)) or (yc1 >= max(yc2,ys2) and yk1 >= max(yc2,ys2))) and xk1==xc2:
            # print('L')
            return True
    if (yc1-yk1)*(yc2-yk2) > 0:
        if not((yc1 <= min(yc2,yk2) and yk1 <= min(yc2,yk2)) or (yc1 >= max(yc2,yk2) and yk1 >= max(yc2,yk2))) and xk1==xk2:
            # print('M')
            return True

    return False



# (up,left,down,right) = (0,1,2,3)
def objFunct(mappings, dir, cores, io_array, dims, graphs):
    """
    Objective function to evaluate the performance of a given mapping of cores to IO operations.
    Args:
        mappings (list): A list of mappings where each mapping defines the allocation of cores to IO operations.
        dir (list): A list where elements are tuples of (y, x) coordinates.
        cores (list): A list of core objects, each containing attributes like scan and patterns.
        io_array (list): A list of IO operations, where each IO operation is defined by a source and sink.
        dims (tuple): Dimensions of the network or system.
        graphs (list): A list of graph structures representing the network topology.
    Returns:
        torch.Tensor: A tensor containing the penalty values for the given mappings.
        list: A list of orders representing the sequence of core allocations.
    Notes:
        - The function calculates the penalty based on the maximum global time required for the IO operations.
        - It uses a priority queue to manage passive cores and resolve conflicts.
        - The function checks for path conflicts between cores to ensure efficient allocation.
    """
        # Step 1: Calculate the number of samples per core mapping
    num_samples = len(mappings) // len(cores) + 1

    # Step 2: Initialize lists to hold penalties and allocation orders
    penalty = []  # Stores penalties for each mapping
    orders = []   # Stores the order of core allocations for each mapping

    # Step 3: Iterate through each mapping in mappings
    for k, maps in enumerate(mappings):
        # Step 4: Prepare io_data to group cores by their associated IO paths
        io_data = [[] for _ in range(len(io_array))]
        for i, m in enumerate(maps):
            io_data[m].append(i)

        # Step 5: Initialize various arrays to track timings and allocation status
        start_time = [0] * len(maps)   # Step 8
        global_time = [0] * len(io_array)  # Step 7
        allocated = [False] * len(maps)  # Tracks whether a core has been allocated
        in_passive = [False] * len(maps)  # Tracks if a core is in the passive queue
        test_times = [[] for _ in range(len(io_array))]  # Step 9

        # Step 10: Calculate test times for each core in io_data
        for j in range(len(io_data)):
            for c in range(len(io_data[j])):
                core = io_data[j][c] + 1
                io = io_array[int(maps[core - 1])]
                src = io[0] - 1
                sink = io[1] - 1
                hsc = hop_counts(graphs, src + 1, core)  # Hop count source to core
                hck = hop_counts(graphs, core, sink + 1)  # Hop count core to sink
                l = cores[k // num_samples][j].scan  # Core's scanning capability
                p = cores[k // num_samples][j].patterns  # Core's pattern property
                time = (max(hsc, hck) + l) * p + (min(hsc, hck) + l - 1)  # Step 11
                test_times[j].append(time)

        # Step 13: Initialize allocation and passive priority queue
        alloc = []  # Stores allocated cores and their end times
        pass_arr = PriorityQueue()  # Passive queue for unallocated cores

        # Step 14: Main loop for allocation while there are unallocated cores
        while (False in allocated):
        # if (len(pass_arr) == 0) or is_passive(maps, global_time, start_time, pass_arr):
        # print(f"Global time: {global_time}")
        # print(f"Start time: {start_time}")
        # print(f"Current allocations: {alloc}")
        # print(f"Passive Queue: {pass_arr.queue}")
            temp_passive = []  # Temporary list for resolving conflicts in the passive queue

            # Step 15: Process cores in the passive queue
            while not pass_arr.empty():
                strt, core = pass_arr.get()
                if allocated[core]:
                    continue

                is_conflict = False  # Flag to detect path conflicts
                io_idx = maps[core]
                core_in_io_indx = io_data[io_idx].index(core)

                for a, all in enumerate(alloc):
                    if start_time[core] < all[1]:
                        # Step 16: Check for path conflicts with already allocated cores
                        core1 = core
                        src1 = io_array[io_idx][0] - 1
                        sink1 = io_array[io_idx][1] - 1
                        core2 = all[0]
                        src2 = io_array[maps[all[0]]][0] - 1
                        sink2 = io_array[maps[all[0]]][1] - 1
                        is_conflict = check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2)
                        if is_conflict:
                            temp_passive.append(core)
                            start_time[core] = max(start_time[core], global_time[maps[all[0]]])
                            break

                # Step 16 continued: Allocate core if no conflicts
                if not is_conflict:
                    allocated[core] = True
                    alloc.append((core, start_time[core] + test_times[io_idx][core_in_io_indx]))
                    global_time[io_idx] = start_time[core] + test_times[io_idx][core_in_io_indx]

                    # Update start times for other cores in the same io_data
                    for c in io_data[io_idx]:
                        if not allocated[c]:
                            start_time[c] = max(start_time[c], global_time[io_idx])

                    # Resolve conflicts for cores in temp_passive
                    for conf in temp_passive:
                        start_time[conf] = max(start_time[conf], global_time[io_idx])

                    break

            # Step 17: Process remaining unallocated cores in io_data
            for idx, io in enumerate(io_data):
                temp_conflict = []
                for indx, c in enumerate(io):
                    if allocated[c] or in_passive[c]:
                        continue

                    is_conflict = False
                    for a, all in enumerate(alloc):
                        if start_time[c] < all[1]:
                            # Step 19: Check conflicts with allocated cores
                            core1 = c
                            src1 = io_array[idx][0] - 1
                            sink1 = io_array[idx][1] - 1
                            core2 = all[0]
                            src2 = io_array[maps[all[0]]][0] - 1
                            sink2 = io_array[maps[all[0]]][1] - 1
                            is_conflict = check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2)
                            if is_conflict:
                                temp_conflict.append(c)
                                start_time[c] = max(start_time[c], global_time[maps[all[0]]])
                                break

                    # Step 20: Allocate core if no conflicts
                    if not is_conflict:
                        allocated[c] = True
                        alloc.append((c, start_time[c] + test_times[idx][indx]))
                        global_time[idx] = start_time[c] + test_times[idx][indx]

                        # Update start times for other cores and resolve conflicts
                        for core in io:
                            if not allocated[core]:
                                start_time[core] = max(start_time[core], global_time[idx])
                        for conf in temp_conflict:
                            start_time[conf] = max(start_time[conf], global_time[idx])
                            pass_arr.put((start_time[conf], conf))
                            in_passive[conf] = True
                        temp_conflict = []
                        break

                    for c in temp_conflict:
                        pass_arr.put((start_time[c], c))
                        in_passive[c] = True

            # Step 15 continued: Refill the passive queue
            temp = []
            while not pass_arr.empty():
                start, core = pass_arr.get()
                temp.append(core)
            for c in temp:
                pass_arr.put((start_time[c], c))

        # Step 23: Append penalty and allocation order
        penalty.append(np.max(global_time) * -1)  # Negate max global time as penalty
        orders.append([item[0] for item in alloc])  # Append allocation order

    # Step 25: Return penalties and allocation orders
    return torch.from_numpy(np.array(penalty)), orders


def rearrange(x:torch.Tensor, beams_buf:torch.Tensor)->torch.Tensor:
    batch_size, num_samples = beams_buf.size()
    # print(f"x: {x.shape}")
    if x.dim() == 1:
        return x.view(batch_size, num_samples).gather(1,beams_buf).view(-1)
    elif x.dim() == 2:
        return x.view(batch_size, num_samples, -1)\
            .gather(1,beams_buf.unsqueeze(-1).expand(-1, -1, x.size(-1)))\
                .view(-1, x.size(-1))
    elif x.dim() == 3:
        return x.permute(1,0,2)\
            .view(batch_size, num_samples, x.size(0), x.size(2))\
            .gather(1,beams_buf.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(0), x.size(2)))\
                .view(-1, x.size(0), x.size(2))\
                    .permute(1,0,2)\
                        .contiguous()

def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        torch.log(-torch.expm1(-torch.exp(-x)))  # Hope for the best
    )


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    CHECK_VALIDITY = False
    if CHECK_VALIDITY:
        g_inv = _shift_gumbel_maximum(g, Z, dim)
        assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
    g[phi == -float('inf')] = -float('inf')
    return g, argmax


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))

"""# Model Classes"""

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                p):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                        dropout=p).to(device)

    def forward(self, input, mask):
        # input shape: (seq_len, batch, input_dim)
        # mask shape: (batch, seq_len)
        # print(input.shape, 'encoder input shape')
        # print(mask.shape, 'encoder mask shape')
        lengths = mask.sum(dim=1)
        # lengths shape: (batch)
        packed_inputs = pack_padded_sequence(input, lengths.to('cpu'), enforce_sorted=False).to(device)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        # output shape: (seq_len, batch, hidden_dim)
        # hidden shape: (n_layers, batch, hidden_dim) cell shape: (n_layers, batch, hidden_dim)
        # if encoder is bidirectional
        # pass hidden and cell through a linear layer to match decoder hidden and cell dimension
        seq_len = input.shape[0]
        # reshape output to (seq_len, batch, hidden_dim)
        return output, (hidden, cell)

class Decoder(nn.Module):
    # module for single step of decoding process
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super(Decoder, self).__init__()
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, hidden, cell):
        # input shape: (batch, input_dim)
        # hidden shape: (n_layers, batch, hidden_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim*2)
        # output shape: (batch, input_dim)
        input = input.unsqueeze(0).to(device)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, logit_clipping=True, clip_value=10):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, dec_dim),
            nn.Tanh(),
        ).to(device)
        self.v = nn.Linear(dec_dim, 1, bias=False).to(device)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

    def forward(self, decoder_output, encoder_outputs, num_io, mask):
        # decoder_output is the output of the decoder of single step
        # encoder_outputs is a list of all the encoder outputs
        # decoder_output shape: (batch, dec_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim)
        # print(decoder_output.shape, " decoder shape")
        # print(encoder_outputs.shape, " encoder shape")
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1).to(device)
        encoder_outputs = encoder_outputs.permute(1, 0, 2).to(device)
        # decoder_output shape: (batch, seq_len, dec_dim)
        # encoder_outputs shape: (batch, seq_len, enc_dim)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2)).to(device)
        # energy shape: (batch, seq_len, dec_dim)
        attention = self.v(energy).squeeze(2).to(device)
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
        # use mask to remove the attention weights for padded values
        # print(attention.shape, 'attention')
        # print(mask.shape, 'mask')
        attention = attention.masked_fill(mask == 0, float('-inf'))
        # attention shape: (batch, seq_len)
        attentions, _ = torch.topk(attention, k=num_io, dim=1)
        attentions = attentions.to(device)
        return attentions

class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device, logit_clipping=True, decoding_type = 'sampling'):
        # print("using lstm")
        super().__init__()
        self.encoder: Encoder = Encoder(input_dim, hidden_dim, n_layers, p)
        self.decoder: Decoder = Decoder(input_dim, hidden_dim, n_layers, p)
        # self.decoder_tsv: Decoder_tsv = Decoder_tsv(input_dim, hidden_dim, n_layers, p)
        self.attn: Attention = Attention(hidden_dim, hidden_dim, logit_clipping)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))
        self.device = device
        self.decoding_type = decoding_type

    def preprocess(self, input, mask) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=1, dtype=torch.long)
        # seq_len is the length of the longest sequence in the batch
        seq_len = torch.max(lengths).item()
        # reshape input and mask to remove extra padding
        # first dimension of the input is seq_len
        input = input[:seq_len]
        # second dimension of the mask is seq_len
        mask = mask[:, :seq_len]
        return input, mask


    def forward(self, input: Tensor, num_io, mask, num_samples=1):

        # sample n_samples mapping solutions for each sequence in the batch
        # input, mask = self.preprocess(input, mask)
        # input shape: (seq_len, batch_size, input_dim)
        batch_size = input.size(1)
        seq_len = input.size(0)
        # print("inp shape",input.shape)
        # Tensor to store the predicted mapping
        # predicted_mappings shape: (batch * num_samples, seq_len)
        predicted_mappings = torch.zeros(batch_size*num_samples, seq_len, dtype=torch.int64).to(self.device)
        # print(mask.shape, 'ptr mask shape')
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        # first input should be a part of model learnable parameters
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        # mask to be used while calculating attention weights
        mask_decoding = mask.clone()
        log_probs_sum = torch.zeros(batch_size*num_samples, dtype=torch.float32).to(self.device)
        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            # logits shape: (batch * num_samples, seq_len)
            log_probs = F.log_softmax(logits, dim=1)
            # print(f"logits: {logits}")
            # print(f"log_probs shape: {log_probs}")
            # log_probs shape: (batch * num_samples, seq_len)
            if t == 0:
                # log_probs shape: (batch , seq_len)
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                    # selected_indices shape: (batch * num_samples,)
                        scores = log_probs + gumbel_like(log_probs)
                        # scores shape: (batch * num_samples, seq_len)
                        # print(f"scores shape: {scores.shape}")
                    elif self.decoding_type == 'greedy':
                        scores = log_probs
                    _, selected_indices = torch.topk(scores, min(num_samples, scores.size(-1)), dim=-1)
                    # selected_indices shape: (batch, min(num_samples, seq_len))
                    if num_samples > log_probs.size(1):
                        # pad second dimension with -1 so that the shape becomes (batch, num_samples)
                        selected_indices = F.pad(selected_indices, (0, num_samples - scores.size(1)), 'constant', -1)
                    selected_indices = selected_indices.view(-1)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples, replacement=True).long().view(-1)
                log_probs = log_probs.repeat_interleave(num_samples, dim=0)
                # print("lp1",log_probs.shape)
                # make log_probs -inf for the values which are -1 in selected_indices
                log_probs = log_probs.masked_fill((selected_indices == -1).unsqueeze(-1), float('-inf'))
                # print("lp2", log_probs.shape)
                if self.decoding_type == 'sampling-w/o-replacement':
                    scores = scores.repeat_interleave(num_samples, dim=0)
                    scores = scores.masked_fill((selected_indices == -1)\
                        .unsqueeze(-1), float('-inf'))
                mask_decoding = mask_decoding.repeat_interleave(num_samples, dim=0)
                # mask_decoding shape: (batch * num_samples, seq_len)
                hidden = hidden.repeat_interleave(num_samples, dim=1)
                cell = cell.repeat_interleave(num_samples, dim=1)
                encoder_outputs = encoder_outputs.repeat_interleave(num_samples, dim=1)
                # selected_indices = selected_indices % 5
            else:
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores, _ = gumbel_with_maximum(log_probs + log_probs_sum.unsqueeze(-1), g_log_probs)
                        # print(f"log_probs: {log_probs.shape}")
                        # print(f"log_probs_sum: {log_probs_sum.shape}")
                        # print(f"scores: {scores.shape}")
                    elif self.decoding_type == 'greedy':
                        scores = log_probs + log_probs_sum.unsqueeze(-1)
                    scores_per_batch = scores.view(batch_size, -1) # (64, 129)
                    # print(scores_per_batch.shape, 'scores shape')
                    top_scores, indices_buf = torch.topk(scores_per_batch, num_samples, dim=1)
                    # print("num_io: ",num_io)
                    # print("indices_buf before: ",indices_buf)
                    # print("indices_buf shape: ", indices_buf.shape)
                    # print("top_scores", top_scores.shape)
                    # indices_buf shape: (batch, min(num_samples, scores.size(1)))
                    # Reshape scores to (batch_size, num_samples * seq_len), but restrict to the first num_io positions
                    beams_buf = torch.div(indices_buf, num_io, rounding_mode='floor')
                    #print(beams_buf, 'beams')
                    indices_buf = indices_buf.fmod(num_io)
                    # indices_buf = indices_buf[:,-1]
                    # print("indices_buf: ", indices_buf)
                    selected_indices = indices_buf.view(-1)
                    predicted_mappings[:,:t] = rearrange(predicted_mappings[:,:t], beams_buf)
                    log_probs_sum = rearrange(log_probs_sum, beams_buf)
                    mask_decoding = rearrange(mask_decoding, beams_buf)
                    log_probs = rearrange(log_probs, beams_buf)
                    hidden = rearrange(hidden, beams_buf)
                    cell = rearrange(cell, beams_buf)
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores = rearrange(scores, beams_buf)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples).long().squeeze(1)
            predicted_mappings[:, t] = selected_indices
            # print(f"t={t}, sel={selected_indices}")
            gather_indices = selected_indices.unsqueeze(-1).clone()
            gather_indices[gather_indices == -1] = 0
            # print(f"log_probs: {log_probs}")
            # print(selected_indices, 'selected')
            # print(f"gather: {gather_indices}")
            # print(f"mask: {mask}")
            curr_log_probs = log_probs.gather(1, gather_indices).squeeze(-1) \
                * mask.repeat_interleave(num_samples, dim=0)[:, t]
            # print(f"curr_log_probs: {curr_log_probs}")
            log_probs_sum += curr_log_probs
            # print(f"log_probs_sum: {log_probs_sum}")
            if self.decoding_type == 'sampling-w/o-replacement':
                # gumbel perturbed log probabilities of partial sequences
                g_log_probs = scores.gather(1, gather_indices).squeeze(-1) \
                    * mask.repeat_interleave(num_samples, dim=0)[:, t]
            # decoder_input shape: (batch, input_dim)
            decoder_input = input.repeat_interleave(num_samples, dim=1)[gather_indices.squeeze(-1), torch.arange(batch_size * num_samples)]
            # update the mask_decoding to remove the pointed inputs
            mask_decoding.scatter_(1, gather_indices, 0)
        # assign -1 to the mappings corresponding to the padded values to denote that it is invalid
        predicted_mappings = predicted_mappings.masked_fill(mask.repeat_interleave(num_samples, dim=0) == 0, -1)
        predicted_mappings = predicted_mappings.view(batch_size, num_samples, seq_len).transpose(0, 1).reshape(-1,seq_len)
        log_probs_sum = log_probs_sum.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
        if self.decoding_type == 'sampling-w/o-replacement':
            g_log_probs = g_log_probs.view(batch_size, num_samples).transpose(0, 1).reshape(-1)
            return predicted_mappings, log_probs_sum, g_log_probs
        else:
            return predicted_mappings, log_probs_sum

"""# Training Model"""

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def epoch_train(data, cores, io_pairs, dims, graphs, model, optimizer, sched, num_samples, best_cost, best_mapping, best_order):
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    # print(len(io_pairs))
    mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=num_samples+1)
    mappings = mappings[:-1, :].to(device)
    # print(f"Mappings: {mappings}")
    threshold = g_log_probs[-1].to(device)
    log_probs = log_probs[:-1].to(device)
    g_log_probs = g_log_probs[:-1].to(device)
    penalty, orders = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs)
    penalty = penalty.to(device)
    # take the value of the last sample for threshold
    log_q = gumbel_log_survival(threshold - g_log_probs).detach().to(device)
    log_importance = (log_probs - log_q).detach().to(device)
    wi_s = ((log_importance.unsqueeze(1).repeat(1,log_importance.size(0)) - log_importance.unsqueeze(0)).exp().sum(dim=0) - 1 + torch.exp(log_q)).to(device)
    importance_normalized = F.softmax(log_importance, dim=0).to(device)
    b_s = torch.sum( importance_normalized * penalty).to(device)
    min_penalty = torch.argmax(penalty)
    if penalty[min_penalty] > best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = mappings[min_penalty]
        best_order = orders[min_penalty]
    # print("Penalty:",penalty)
    # print("Baseline:",b_s)
    # print("pen - b_s: ",penalty-b_s)
    # print("log_importance: ", log_importance)
    # print("Weights: ",wi_s)
    # print("Log Probs: ",log_probs)
    loss = torch.sum((1/wi_s) * log_probs * (penalty - b_s)).to(device)
    # loss = (loss+1e5).to(device)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
    optimizer.step()
    sched.step(loss)

    return penalty, min_penalty, loss.item(), best_cost, best_mapping, best_order

def epoch_rl(data, cores, io_pairs, dims, graphs, model, optimizer, num_samples, best_cost, best_mapping, alpha):
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    n = num_samples//data.size(1)
    b = 0
    losses = []
    for i in tqdm(range(n)):
        mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=2)
        mappings = mappings[:-1, :].to(device)
        threshold = g_log_probs[-1].to(device)
        log_probs = log_probs[:-1].to(device)
        g_log_probs = g_log_probs[:-1].to(device)
        penalty, order = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs)
        penalty = penalty.to(device)
        #print("Order  = ", order)
        # print(f"Mappings: {len(mappings)}")
        # print(f"pen: {len(penalty)}")
        log_q = gumbel_log_survival(threshold - g_log_probs).detach().to(device)
        log_importance = (log_probs - log_q).detach().to(device)
        wi_s = ((log_importance.unsqueeze(1).repeat(1,log_importance.size(0)) - log_importance.unsqueeze(0)).exp().sum(dim=0) - 1 + torch.exp(log_q)).to(device)
        importance_normalized = F.softmax(log_importance, dim=0).to(device)
        b_s = torch.sum( importance_normalized * penalty).to(device)
        min_penalty = torch.argmax(penalty)
        if penalty[min_penalty] > best_cost:
            best_cost = penalty[min_penalty]
            best_mapping = mappings[min_penalty]
        loss = torch.sum((1/wi_s) * log_probs * (penalty - b)).to(device)
        print(f"RL Loss = {loss},  Best Cost = {best_cost}, Best mapping = {best_mapping}")
        # print(f"Best Mapping: {best_mapping}")
        # print(f"Best Cost: {best_cost}")
        # print(f"threshold: {threshold}")
        # print(f"imp_norm: {importance_normalized}")
        # print(f"g_log_probs: {g_log_probs}")
        # print(f"log_q: {log_q}")
        # print("Penalty:",penalty)
        # print("Baseline:",b_s)
        # print("pen - b_s: ",penalty-b_s)
        # print("log_importance: ", log_importance)
        # print("Weights: ",wi_s)
        # print("Log Probs: ",log_probs)
        # print(f"Loss: {loss}")
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        # sched.step(loss)
        b = alpha*b + (1-alpha)*b_s.mean(dim=0)

    return best_cost, best_mapping, losses


def model_train(numcore, model, optimizer, data, cores, io_pairs, dims, graphs, num_samples, batch_size, lr, wt_decay, factor, patience, dropout, n_layers, input_dim, hidden_dim, MAX_EPOCH):
    pens = []
    losses = []
    # print("i am in model_train")
    best_cost = float('-inf')
    best_mapping = torch.from_numpy(np.zeros(len(cores))).to(device)
    best_order = list(range(len(cores)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    # samples = generateSamples(num_samples)
   # samples = generateSamples(num_samples)

    for epoch in tqdm(range(MAX_EPOCH)):
        print('Epoch {}: '.format(epoch+1))

        # model.train(True)
        penalty, min_penalty, loss, best_cost, best_mapping, best_order = epoch_train(data, cores, io_pairs, dims, graphs, model, optimizer, scheduler, num_samples, best_cost, best_mapping, best_order)

        # model.eval()
        # avg_val_loss, curr_pred = epoch_val()

        pens.append(penalty[min_penalty].item())
        losses.append(loss)
        print('Train loss = {}, Train Penalty = {}, Best Cost = {}, Best mapping = {}, Order = {}'.format(loss, penalty[min_penalty].item(), best_cost, best_mapping, best_order))

        if penalty[min_penalty] > best_cost:
            count_not_decrease += 1
        else:
            count_not_decrease = 0
        if count_not_decrease >= 1000:
            print('Early stopping at epoch {}'.format(epoch))
            break
    epochs = range(1, len(pens) + 1)

# Plot the losses against the indices
    np.save(f"pens_{numcore}cores.npy", pens)
    #plt.plot(epochs, pens, label='Penalty')
    #plt.xlabel('Epochs')
    #plt.ylabel('Penalty')
    #plt.title('Penalty Curve')
    #plt.legend()
    #plt.savefig(f'penalty_curve{idx}.png')
    #prints(pens)

    np.save(f'loss_{numcore}cores.npy', losses)
    #plt.plot(epochs, losses, label='Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Loss Curve')
    #plt.legend()
    #plt.savefig(f'loss_curve{idx}.png')
    #print(losses)

    return pens, losses, best_cost, best_mapping

def model_rl(numcore, model, optimizer, alpha, data, cores, io_pairs, dims, graphs, num_samples, batch_size, lr, wt_decay, factor, patience, dropout, n_layers, input_dim, hidden_dim, MAX_EPOCH):
    best_cost = float('-inf')
    best_mapping = torch.from_numpy(np.zeros(len(cores))).to(device)
    # alpha = 0.99
    model.load_state_dict(torch.load('ordered_active_search_56cores.pt'))
    optimizer.load_state_dict(torch.load('ordered_opt_56cores.pt'))
    best_cost, best_mapping, losses = epoch_rl(data, cores, io_pairs, dims, graphs, model, optimizer, num_samples, best_cost, best_mapping, alpha)
    losses = np.array(losses)
    np.save(f'loss_rl_{numcore}cores_{alpha}.npy', losses)
    # print(f"Best Mapping: {best_mapping}")

def force_cudnn_initialization():
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def test(numcore, model, optimizer, alpha, data, cores, io_pairs, dims, graphs, num_samples, batch_size, lr, wt_decay, factor, patience, dropout, n_layers, input_dim, hidden_dim, MAX_EPOCH):
    #model.load_state_dict(torch.load('active_search_rl_32cores.pt'))
    #optimizer.load_state_dict(torch.load('opt_32cores.pt'))
    model.eval()
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=1)
    penalty = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs).to(device)
    return torch.max(penalty).item(), mappings


"""# Driver Code"""
def main(args):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # device = torch.device('cpu')
    # torch.cuda.empty_cache()
    # force_cudnn_initialization()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
   # loss_list = torch.load(args.loss_list_path) if args.loss_list_path else []
    #mpnn_ptr, dataloader = initialize(args.model_path, args.datapath, args.batch_size, device)
    mode = args.mode
    if mode is 'train':
        #---TRAIN---

        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        model.load_state_dict(torch.load(f'ordered_active_search_48cores.pt'))
        optimizer.load_state_dict(torch.load(f'ordered_opt_48cores.pt'))
        numcore=[56]  # numcore = [7,10,14,28,32]
        for c in numcore:
            data, cores, io_pairs, dims, graphs = prep_data(c, False)
            pens_list, loss_list, best_cost, best_mapping = model_train(c, model, optimizer, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            torch.save(model.state_dict(), f'ordered_active_search_{c}cores.pt')
            torch.save(optimizer.state_dict(), f'ordered_opt_{c}cores.pt')
        #---TRAIN---

    elif mode is 'rl':
        #---RL---

        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        numcore = [args.num_cores]
        for c in numcore:
            data, cores, io_pairs, dims, graphs = prep_data(c, False)
            # print(f"io = {len(io_pairs)}")
            model_rl(c, model, optimizer, args.alpha, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            torch.save(model.state_dict(), f'ordered_active_search_rl_{c}cores_{args.alpha}.pt')
            torch.save(optimizer.state_dict(), f'ordered_opt_rl_{c}cores_{args.alpha}.pt')

        #---RL---

    elif mode is 'test':
        #---TEST---

        numcore = [args.num_cores]
        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        model.load_state_dict(torch.load(f'fixed_active_search_rl_{args.num_cores}cores_0.99.pt'))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        optimizer.load_state_dict(torch.load(f'fixed_opt_rl_{args.num_cores}cores_0.99.pt'))
        data, cores, io_pairs, dims, graphs = prep_data(args.num_cores, True)
        costs = []
        mappings = []
        for i in tqdm(range(args.num_test_iter)):
            cost, mapp = test(args.num_cores, model, optimizer, args.alpha, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            costs.append((-1)*cost)
            mapp = mapp[0]
            mappings.append([x.item() for x in mapp])

        # print(np.max(costs))
        # print(np.min(costs))
        # idx = np.argmin(costs)
        print(mappings)

        #---TEST---

    else:
        print("Wrong mode argument")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train/rl/test", type=str, default='test')
    parser.add_argument("--num_samples", help="number of samples", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-8)
    #parser.add_argument("--model_path", help="path to pretrained model", type=str, default=None)
    #parser.add_argument("--loss_list_path", help="path to loss list", type=str, default=None)
    parser.add_argument("--wt_decay", help="decay rate for regularization", type=float, default=1e-5)
#og value = 1e-5
    parser.add_argument("--factor", help="factor", type=float, default=0.001)
#og value = 0.5
    parser.add_argument("--patience", help="patience", type=float, default=100)
#og value = 5
    parser.add_argument("--dropout", help="dropout", type=float, default=0.1)
#og value  =0.1
    parser.add_argument("--n_layers", help="num_layer_encoder", type=int, default=2)
    parser.add_argument("--input_dim", help="input_dim_enc", type=int, default=2)
    parser.add_argument("--hidden_dim", help="hidden_dim", type=int, default=2)
#og value =1
    parser.add_argument("--epoch", help="MAX_EPOCH", type=int, default=10000)
    parser.add_argument("--num_cores", type=int, default=7)
    parser.add_argument("--num_test_iter", help="number of test iterations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.99)
    args = parser.parse_args()
    main(args)
