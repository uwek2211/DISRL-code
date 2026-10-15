from collections import defaultdict


def int_set(values):
    return {int(value) for value in values}


def normalize_labels(mapping):
    return {str(node): int_set(cids) for node, cids in mapping.items()}


def copy_set_map(mapping, cast):
    return {
        int(key): {cast(value) for value in values} for key, values in mapping.items()
    }


def merge_set_maps(*mappings):
    merged = {}
    for mapping in mappings:
        for key, values in mapping.items():
            merged.setdefault(int(key), set()).update(values)
    return merged


def discard_label(multi_labels, cid, keep_nodes=()):
    keep_nodes = {str(node) for node in keep_nodes}
    for node, labels in list(multi_labels.items()):
        if cid not in labels or str(node) in keep_nodes:
            continue
        labels = set(labels)
        labels.discard(cid)
        if labels:
            multi_labels[node] = labels
        else:
            multi_labels.pop(node, None)


def build_cid2nodes(multi_labels, nodes_in_graph=None, ordered=False):
    nodes_in_graph = set(multi_labels) if nodes_in_graph is None else nodes_in_graph
    cid2nodes = defaultdict(list if ordered else set)
    for node, cids in multi_labels.items():
        if node in nodes_in_graph:
            for cid in cids:
                group = cid2nodes[int(cid)]
                (group.append if ordered else group.add)(str(node))
    return cid2nodes
