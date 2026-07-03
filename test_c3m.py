import collections

class Agent:
    def __init__(self):
        self.tracking_data = collections.defaultdict(list)

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data[tag].append(value)

agent = Agent()
agent.track_data("Loss / C3M/loss/loss", 1.234)

def _last(key):
    v = agent.tracking_data.get(key, [float("nan")])
    return v[-1] if isinstance(v, list) else float(v)

print(f"{_last('Loss / C3M/loss/loss'):.3g}")
