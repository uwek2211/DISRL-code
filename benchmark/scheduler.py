"""Time-window scheduler for evolution rules and stateful events."""


class Scheduler:
    """Schedule rule-level operations and event-level transformations."""

    def __init__(self):
        self.rule_windows = []
        self.events = []

    def add_rule_window(self, start, end, rules):
        """Enable rule objects in the half-open interval [start, end)."""
        if not isinstance(rules, (list, tuple)):
            raise TypeError("rules must be a list or tuple of rule objects")
        if start >= end:
            raise ValueError("start must be smaller than end")
        self.rule_windows.append((int(start), int(end), list(rules)))

    def get_active_rules(self, t):
        """Return rule objects active at timestep t."""
        t = int(t)
        active = []
        for left, right, rules in self.rule_windows:
            if left <= t < right:
                active.extend(rules)
        return active

    def add_event(self, event):
        """Add a stateful event object with active/on_start/step methods."""
        self.events.append(event)

    def get_active_events(self, t):
        """Return event objects active at timestep t."""
        t = int(t)
        return [event for event in self.events if event.active(t)]
