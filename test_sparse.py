import torch
import torch_sparse

rowptr = torch.tensor([0, 2, 3], dtype=torch.long)
col = torch.tensor([1, 0, 1], dtype=torch.long)

out = torch.ops.torch_sparse.partition(
    rowptr,
    col,
    None,
    2,      # num_parts
    False   # recursive
)
print(out)