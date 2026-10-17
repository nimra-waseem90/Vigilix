import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion.event_format import create_event

event = create_event(
    event="weapon",
    label="Gun",
    confidence=0.91,
    timestamp=15.4
)

print("\nVIGILIX EVENT")
print(event)