from typing import Optional

from rich.tree import Tree
from rich import print

from .type_aliases import TreeNode


def build_rich_tree(
        data: TreeNode,
        parent: Optional[Tree] = None
  ) -> None:
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
