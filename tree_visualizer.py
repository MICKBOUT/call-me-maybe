from rich.tree import Tree
from rich import print


def build_rich_tree(data: dict, parent=None):
    for key, value in data.items():
        if key == 'name':
            continue
        label = f"[cyan]{key}[/] [dim]{value['name']}[/]"
        branch = parent.add(label) if parent else Tree(label)
        build_rich_tree(value, branch)
        if parent is None:
            print(branch)
