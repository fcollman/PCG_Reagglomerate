"""Module for agglomerating graphs"""
import numpy as np
from scipy.sparse import coo_matrix
import pqdict
import pandas as pd


class HeapEdge(object):
    """HeadEdge class to store edges in priority queue"""

    def __init__(self, edge: tuple, affinity: float, area: float):
        """a heap edge object

        Args:
            edge (tuple): (v0:int, v1:int) the edge to store (v0<v1)
            affinity (float): the affinity of the edge
            area (float): the area of this edge
        """
        self.edge = edge
        self.affinity = affinity
        self.area = area

    def __repr__(self):
        return f"Node weight: {self.affinity}, edge: {self.edge}"

    def __lt__(self, other):
        return (self.affinity) < (other.affinity)


def robust_pop(pq, v0, v1):
    """Remove the v0 to v1 edge from the priority queue

    Args:
        pq (pqdict.maxpq()): _description_
        v0 (int): the index of the v0 vertex in this edge
        v1 (int): the index of the v1 vertex in this edge

    Raises:
        KeyError: If this edge doesn't exist in queue (will check for v0,v1 and v1,v0)

    Returns:
        HeapEdge: the edge stored in the priority queue heap
    """
    # This protects against the cases where the edges were specified twice
    # or we go looking of a,b but only b,a is in the queue
    if v0 < v1:
        he1 = pq.pop((v0, v1), None)
        if he1 is None:
            he1 = pq.pop((v1, v0), None)
    else:
        he1 = pq.pop((v1, v0), None)
        if he1 is None:
            he1 = pq.pop((v0, v1), None)
    if he1 is None:
        raise KeyError(f"{v0},{v1} does not exist in queue")
    return he1


def remap_edges(edges, affinities, areas):
    """renumber edge vertex indices to the lowest possible set of integers
    while removing duplicate edges, and deterministically ordering them.
    Assumes if an edge is listed twice that its properties are identical.

    Args:
        edges (np.array): Nx2 list of edges
        affinities (np.array): N long vector of affinities
        areas (np.array): N long vector of areas

    Returns:
        a tuple containing
        unique_edges (np.array): a re-indexed set of edges
        vertex_map: old_ids to new_ids dictionary
    """
    # Create a set of all the vertex indices in the edges list
    vertex_set = set(edges.flatten())

    # Create a mapping from the old vertex indices to the new indices
    vertex_map = {old: new for new, old in enumerate(vertex_set)}

    # Remap the vertex indices in the edges array
    remapped_edges = np.array([(vertex_map[v0], vertex_map[v1]) for v0, v1 in edges])

    # Sort each row in ascending order
    remapped_edges = np.sort(remapped_edges, axis=1)

    # Remove duplicate rows
    _, unique_indices = np.unique(remapped_edges, return_index=True, axis=0)
    unique_edges = remapped_edges[unique_indices]

    return (
        unique_edges,
        vertex_map,
        np.array(list(vertex_set)),
        affinities[unique_indices],
        areas[unique_indices],
    )


def agglomerate_graph(edges, affinities, areas, threshold, affinity_merge="mean"):
    """agglomerate a graph, agglomerate components with the highest affinities
    and merge duplicate edges using area weighted mean affinity
    or sum affinity.

    It will remove any duplicate edges and assume that the affinities and
    areas were the same, rather than adding them.

    Args:
        edges (np.array): a Nx2 list of edges
        affinities (np.array): a N long list of affinities
        areas (np.array): a N long list of area values for edges
        threshold float: At what affinity level to stop merging
        affinity_merge (str, optional): How to merge edges.
            Can be 'sum' or 'mean'. Defaults to 'mean'.
            'mean' does area weighted mean affinity,
            'sum' does total affinity.

    Raises:
        ValueError: If there is a loop in the graph

    Returns:
        pd.DataFrame: with columns [v0, v1, merged_affinity, merged_area]
        each row is an edges that represent an agglomeration of components.
        Previous vertices that were merged in are implicitly represented.
        The v0 vertex is merged into the v0 vertex.
        So v1 becomes a stand-in for all vertices in that component.
        All edges from v0 are redirected/merged into v1.

        merged_affinity will represent the sum or area weighted mean affinity
        of that agglomeration event (depending on the choice of affinity_merge)

        merged_area will represent the summed area between the aggomerated components
    """
    # use remap edges to reduce index namespace, and deterministically
    # sort edges, while removing duplicates

    edges, _, vertex_list, affinities, areas = remap_edges(
        edges, affinities=affinities, areas=areas
    )

    # Convert the graph to a sparse matrix representation
    n = np.max(edges) + 1
    print(n)
    rows = np.hstack([edges[:, 0], edges[:, 1]])
    cols = np.hstack([edges[:, 1], edges[:, 0]])
    print(rows.shape)
    graph_matrix = coo_matrix(
        (np.ones(2 * len(affinities)), (rows, cols)), shape=(n, n)
    )
    graph_matrix = graph_matrix.tolil()

    # Initialize disjoint sets data structure
    # not sure if needed
    # sets = [{k} for k in range(n)]

    # Initialize list for storing MST edges
    mst_edges = []

    # Initialize queue for storing edges

    d = {}
    # Add all edges to the heap
    for (i, j), affinity, area in zip(edges, affinities, areas):
        d[(i, j)] = HeapEdge((i, j), affinity, area)

    # initialize a pqdict priority queue heap
    pq = pqdict.maxpq(d)

    iteration = 0
    # calculate the starting degree of every
    # vertex, used to optimize who to collapse
    degree = np.sum(graph_matrix, axis=1)
    degree = np.squeeze(np.asarray(degree))

    while len(pq) > 0:
        # pop the top item off heap
        _, node = pq.popitem()

        # if affinity is below threshold, stop
        if node.affinity < threshold:
            print(node, threshold)
            break

        # progress printing
        iteration += 1
        if (iteration % 1000) == 0:
            print(iteration, len(pq), node.affinity)

        v0 = node.edge[0]
        v1 = node.edge[1]

        if v0 != v1:
            #
            # s0 = sets[v0]
            # s1 = sets[v1]

            # we want to eliminate the node with the smallest degree
            # because it will be more efficient.
            if degree[v0] > degree[v1]:
                vt = v0
                v0 = v1
                v1 = vt

            # if s0 != s1:
            # not sure why this was needed
            mst_edges.append(
                [vertex_list[v0], vertex_list[v1], node.affinity, node.area]
            )

            # v0 is dissapearing from the graph
            # if we were tracking sets we would do this
            # sets[v1] = s1.union(s0)

            # erase the edge e = {v0,v1}
            graph_matrix[v0, v1] = 0
            graph_matrix[v1, v0] = 0

            # grab copy of all the edges involving v0
            non_zero_cols = graph_matrix.rows[v0].copy()

            # loop over other edges {v0,e0}
            for e0 in non_zero_cols:
                # remove this edge from the graph
                graph_matrix[v0, e0] = 0
                graph_matrix[e0, v0] = 0

                if e0 == v0:
                    raise ValueError("loop in the graph")

                # get this edge from the heap
                he1 = robust_pop(pq, v0, e0)

                # if the {v1,e0} edge exists
                if graph_matrix[v1, e0]:
                    # then {v0,e0} and {v1,e0} both exist
                    # and we need to merge them

                    # get the {v1,e0} edge
                    he2 = robust_pop(pq, v1, e0)

                    if affinity_merge == "mean":
                        # calculate it's mean affinity, weighted by area
                        aff = (
                            (he1.affinity * he1.area) + (he2.affinity * he2.area)
                        ) / (he1.area + he2.area)
                    elif affinity_merge == "sum":
                        # alternative is to get it's total affinity
                        aff = he1.affinity + he2.affinity

                    if v1 <= e0:
                        pq[(v1, e0)] = HeapEdge((v1, e0), aff, he1.area + he2.area)
                    else:
                        pq[(e0, v1)] = HeapEdge((e0, v1), aff, he1.area + he2.area)
                    # reduce the degree of e0 by 1, as it has one fewer edges now
                    # v1 doesn't change it's number of edges
                    degree[e0] -= 1
                else:
                    # then we simply replace {v0,e0} with {v1,e0}
                    if v1 <= e0:
                        # we need to replace {v0,e0} with {v1, e0}
                        pq[(v1, e0)] = HeapEdge((v1, e0), he1.affinity, he1.area)
                    else:
                        pq[(e0, v1)] = HeapEdge((e0, v1), he1.affinity, he1.area)
                    # increment the degree of v1 because it has one more edge
                    # v0 had one edge removed and one edge added so its the same
                    degree[v1] += 1

                    #
                    graph_matrix[e0, v1] = 1
                    graph_matrix[v1, e0] = 1
            # v0 degree should now be 0 but we don't need to set it
            # because it will never be reached again
            # degree[v0]=0

    # convert the edges to a dataframe
    df = pd.DataFrame(mst_edges)
    df.columns = ["v0", "v1", "merged_affinity", "merged_area"]
    return df
