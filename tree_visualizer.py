from rich.tree import Tree
from rich import print
from rich.console import Console


def build_rich_tree(data: dict, parent=None):
    for key, value in data.items():
        if key == 'name':
            continue
        label = f"[cyan]{key}[/] [dim]{value['name']}[/]"
        branch = parent.add(label) if parent else Tree(label)
        build_rich_tree(value, branch)
        if parent is None:
            print(branch)


console = Console()


def build_rich_tree_fn(dict_function: dict):
    root = Tree("[bold magenta]Functions[/]")

    for fn_name, fn_data in dict_function.items():
        # Function node
        fn_branch = root.add(f"[bold cyan]fn[/] [bold yellow]{fn_name}[/]")

        # Description
        fn_branch.add(f"[dim]{fn_data.get('description', 'No description')}[/]")

        # Parameters
        params = fn_data.get("parameters", {})
        if params:
            param_branch = fn_branch.add("[bold green]parameters[/]")
            for param_name, param_info in params.items():
                param_branch.add(f"[white]{param_name}[/] : [italic blue]{param_info.get('type', '?')}[/]")

        # Returns
        returns = fn_data.get("returns", {})
        if returns:
            fn_branch.add(f"[bold red]returns[/] : [italic blue]{returns.get('type', '?')}[/]")

    console.print(root)
