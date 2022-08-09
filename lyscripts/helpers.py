"""
This module contains frequently used functions as well as instructions on how
to parse and process the raw data from different institutions
"""
from rich.console import Console

CROSS = "[bold red]✗[/bold red]"
CIRCL = "[bold yellow]∘[/bold yellow]"
CHECK = "[bold green]✓[/bold green]"

class ConsoleReport(Console):
    """
    Small extension to the `Console` class of the rich package.
    """
    def success(self, *objects, **kwargs) -> None:
        """Prefix a bold green check mark to any output."""
        objects = [CHECK, *objects]
        return super().print(*objects, **kwargs)

    def info(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow circle to any output."""
        objects = [CIRCL, *objects]
        return super().print(*objects, **kwargs)

    def failure(self, *objects, **kwargs) -> None:
        """Prefix a bold red cross to anything printed."""
        objects = [CROSS, *objects]
        return super().print(*objects, **kwargs)

report = ConsoleReport()


def get_graph_from_(params_graph: dict):
    """
    Build the graph for the `lymph` model from the graph in the params file. I cannot
    simply write the graph in the params file as I like because YAML does not support
    tuples as keys in a dictionary.
    """
    lymph_graph = {}
    for node_type, node_dict in params_graph.items():
        for node_name, out_connections in node_dict.items():
            lymph_graph[(node_type, node_name)] = out_connections
    return lymph_graph
