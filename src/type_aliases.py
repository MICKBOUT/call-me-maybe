from typing import TypeAlias

# Recursive tree node for function trie
TreeNode: TypeAlias = dict[int | str, "TreeNode | str"]

# Token id
TokenId: TypeAlias = int

# Logits list
Logits: TypeAlias = list[float]

# Constrained set of token ids
ConstrainedSet: TypeAlias = set[int]
