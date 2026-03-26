"""Rich Live event stream for real-time analysis display."""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_EVENT_STYLES = {
    "DETECT": "cyan",
    "DESCRIBE": "dim",
    "IMPACT": "red bold",
    "CONTROL_A": "green",
    "CONTROL_B": "blue",
    "INCOMPLETE": "yellow dim",
    "FILTERED": "dim italic",
}


class EventStream:
    """Rolling table of analysis events displayed via Rich Live."""

    def __init__(self, max_events: int = 15) -> None:
        self._max_events = max_events
        self._events: list[tuple[str, str, str]] = []

    def add(self, timestamp: str, event_type: str, details: str) -> None:
        """Append an event, trimming to max_events."""
        self._events.append((timestamp, event_type, details))
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

    def get_renderable(self) -> Group:
        """Return a Rich Group with a header panel and the event table."""
        table = Table(show_header=True, expand=True, pad_edge=False)
        table.add_column("Time", style="bold", width=8, no_wrap=True)
        table.add_column("Event", width=12, no_wrap=True)
        table.add_column("Details", ratio=1)

        for ts, etype, details in self._events:
            style = _EVENT_STYLES.get(etype, "")
            table.add_row(ts, Text(etype, style=style), Text(details, style=style))

        header = Panel(
            Text("Fighter IQ — Live Analysis", style="bold white"),
            border_style="blue",
            expand=True,
        )
        return Group(header, table)
