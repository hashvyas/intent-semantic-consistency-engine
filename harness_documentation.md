# ISCE V2 Pipeline Validation Harness Documentation

The **ISCE Manual Validation Harness** is a developer-facing debugging and demonstration tool for the research framework. It allows you to inject single V2X messages or coordinate traffic scenarios into the ISCE pipeline, trace them through the full 4-stage security architecture (PKI decryption/decoding → SCSV technical validation → MBD simulated behavioral assessment → CSIA cooperative trust reasoning), profile sub-stage execution times, generate observability and trust graphs, and build regression test suites.

---

## 1. Running the Harness

The manual pipeline driver is located at `manual_pipeline_test.py` in the workspace root.

### Mode A: Single Message Validation
Analyze a single JSON message and display the detailed walkthrough:
```bash
python manual_pipeline_test.py test_messages/benign/normal_car.json
```
For B1 failures:
```bash
python manual_pipeline_test.py test_messages/b1_fail/impossible_speed.json
```

### Mode B: Directory / Batch Execution
Run all JSON files in a directory statefully in chronological order, and display a batch summary at the end:
```bash
python manual_pipeline_test.py test_messages/benign/
```

### Mode C: Scenario Runner
Run a complex coordinate attack scenario. Preserves state between vehicles and displays walkthrough details step-by-step:
```bash
python manual_pipeline_test.py scenarios/sybil/
```

---

## 2. Command Line Options

| Flag | Description |
|---|---|
| `--step` | Enable step-by-step pausing mode. Pauses and prompts the user after every internal pipeline block (B1, Graph, Context, Threshold, Reasoning, Propagation, Final Assessment). |
| `--log` | Save execution run logs to a file in the `logs/` folder. Uses sequential naming: `logs/run_YYYY_MM_DD_NN.json`. |
| `--regression` | Execute all test cases stored in the regression suite (`test_messages/regression/`). |
| `--add-regression <file>` | Copy a failing JSON message file to the regression suite. |

### Step-by-Step Mode Example
Useful for live demonstrations:
```bash
python manual_pipeline_test.py test_messages/benign/normal_car.json --step
```

### Logging Option Example
Saves full execution outputs, latencies, and similarities for audit and forensics:
```bash
python manual_pipeline_test.py test_messages/b2/sybil/ --log
```

---

## 3. Test Message Library

The test message library is located under `test_messages/` and structured as follows:

- `benign/`
  - `normal_car.json` — A standard passenger vehicle CAM.
  - `normal_truck.json` — A heavy commercial truck CAM.
  - `motorcycle.json` — A motorcycle CAM.
  - `rsu_message.json` — Trusted infrastructure road side unit message.
- `b1_fail/`
  - `replay.json` — Duplicate messages checking B1 replay caches.
  - `stale_timestamp.json` — Message with timestamp exceeding freshness tolerance.
  - `malformed.json` — Missing keys and NaN values.
  - `invalid_coordinates.json` — Position coordinates out of bounds.
  - `impossible_speed.json` — Vehicle speed exceeding physical capabilities.
  - `impossible_acceleration.json` — Excessive speed variation per frame.
  - `certificate_switch.json` — Rapid rotation of credentials.
- `b2/`
  - Coordination attacks generated statefully: `sybil/`, `replay/`, `collusion/`, `fabrication/`, and `mixed/` scenarios.
- `context/`
  - Context-aware validation streams: `highway/`, `urban/`, `intersection/`, `tunnel/`, and `roundabout/`.
- `stress/`
  - Density stress test flows: `10_nodes/`, `50_nodes/`, and `100_nodes/`.

---

## 4. Regression Support

To add a failing message (for example, `impossible_speed.json`) to the regression suite:
```bash
python manual_pipeline_test.py --add-regression test_messages/b1_fail/impossible_speed.json
```
To run the regression suite at any time:
```bash
python manual_pipeline_test.py --regression
```

---

## 5. Latency and Graph Output

- **Latency Breakdown**: Profiling metrics are printed at the end of each run, measuring average B1 time, B2 time, and B2 subcomponents (Graph construction, Motion context, Adaptive thresholds, Behavior reasoning, and Trust propagation).
- **PNG Visualization**: If `networkx` and `matplotlib` are installed, the harness automatically generates two plots (`observability_graph_<timestamp>.png` and `trust_graph_<timestamp>.png`) under the `visualizations/` folder showing current node connectivity and color-coded trust levels.
