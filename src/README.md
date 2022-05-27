
### _`src` structure_

This directory contains functions as suppliment to the STTTrkX Pipeline. Intended for working on raw CSV files.

It includes functionality such as

- `drawing.py` - Drawing utilities for events.
- `event.py` - Composing a full 'event' from 'hits', 'tubes', 'particles', 'truth' CSV files.

- `reader.py` - CSV Reader Class
    - read all events
    - compose them together
    - return with necessary data columns

- `utils_dir.py` - Some dir. utilities (similar to stttrkx-iml).
- `utils_math.py` - Some math utilities.
- `stt.csv` - Detector file for STT

- A.O.B
