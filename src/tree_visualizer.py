from typing import Optional

from rich.tree import Tree
from rich import print

from .type_aliases import TreeNode


def build_rich_tree(
        data: TreeNode,
        parent: Optional[Tree] = None
  ) -> None:
    """Builds and visualizes a tree structure using the Rich library.

    Recursively constructs a tree from a nested dictionary structure, skipping
    the 'name' key and only processing dictionary values. Each node is labeled
    with the key and the 'name' value from the dictionary. If no parent is
    provided, the root tree is printed to the console.

    Args:
        data (TreeNode): A dictionary-like object representing the tree node,
            containing keys and nested dictionaries.
        parent (Optional[Tree]): The parent Tree node from the Rich library.
            If None, a new root Tree is created and printed.

    Returns:
        None: This function does not return a value; it modifies and prints
            the tree structure in place.
    """
    for key, value in data.items():
        if key == 'name':
            continue
        if not isinstance(value, dict):
            continue
        label = f"[cyan]{key}[/] [dim]{value['name']}[/]"
        branch: Tree = parent.add(label) if parent else Tree(label)
        build_rich_tree(value, branch)
        if parent is None:
            print(branch)
