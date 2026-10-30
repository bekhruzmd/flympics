# Flympics brain: UML

These diagrams show how the connectome-derived controller in `brain/` drives the
100 m sprint, and how training and the tests run.

## 1. Class diagram

`FlyBrainController` is one of three controllers behind the `Controller`
interface. `Sprint` sees only observations and actions, so it never touches the
neural model.

```mermaid
classDiagram
    direction LR

    class OlympicEvent {
        <<abstract>>
        +reset(seed) observation
        +observe() observation
        +step(action) observation
        +is_finished() bool
        +score() float
    }
    class Sprint {
        +distance = 100.0
        +dt = 0.1
        +max_time = 60.0
        +acceleration = 1.25
        +drag = 0.16
        +position
        +velocity
        +time
        +steps
    }
    OlympicEvent <|-- Sprint

    class Controller {
        <<abstract>>
        +reset(seed)
        +act(observation) str
    }
    class RandomController
    class RuleController {
        +target_velocity = 7.0
    }
    class FlyBrainController {
        +circuit: ConnectomeCircuit
        +encoder: SensoryEncoder
        +network: RateNetwork
        +decoder: MotorDecoder
        +last_summary: dict
        +trained(path)$ FlyBrainController
        +reset(seed)
        +act(observation) str
    }
    Controller <|-- RandomController
    Controller <|-- RuleController
    Controller <|-- FlyBrainController

    class ConnectomeCircuit {
        <<dataclass, frozen>>
        +neuron_ids: tuple
        +roles: tuple
        +cell_types: tuple
        +weights: csr_matrix (post x pre)
        +size int
        +indices(role) ndarray
    }
    class SensoryEncoder {
        +indices: sensory neurons
        +sprint_distance = 100.0
        +encode(observation) stimulus
    }
    class RateNetwork {
        +leak = 0.35
        +gain = 3.0
        +state: ndarray(135)
        +reset()
        +step(stimulus, substeps=2) activity
    }
    class MotorDecoder {
        +extensor: indices
        +flexor: indices
        +threshold
        +decode(activity) action, summary
    }

    FlyBrainController *-- ConnectomeCircuit
    FlyBrainController *-- SensoryEncoder
    FlyBrainController *-- RateNetwork
    FlyBrainController *-- MotorDecoder
    SensoryEncoder ..> ConnectomeCircuit : reads sensory indices
    RateNetwork ..> ConnectomeCircuit : uses weights
    MotorDecoder ..> ConnectomeCircuit : reads motor indices
    MotorDecoder ..> Sprint : ACCELERATE / COAST

    class load_manc_front_leg_circuit {
        <<function>>
        reads data/manc_front_leg_v1/neurons.csv and edges.csv
    }
    load_manc_front_leg_circuit ..> ConnectomeCircuit : creates
```

## 2. Sequence diagram: one sprint step

`experiments/run_sprint.py::run()` repeats this loop until the runner reaches
100 m or 60 s passes (at most 600 steps of 0.1 s).

```mermaid
sequenceDiagram
    autonumber
    participant Run as run_sprint.run()
    participant Ev as Sprint
    participant Ctl as FlyBrainController
    participant Enc as SensoryEncoder
    participant Net as RateNetwork
    participant Dec as MotorDecoder

    Run->>Ev: reset(seed)
    Run->>Ctl: reset(seed)
    Ctl->>Net: reset() (state = 0)

    loop until Ev.is_finished()
        Run->>Ev: observe()
        Ev-->>Run: {distance_to_finish, velocity}
        Run->>Ctl: act(observation)
        Ctl->>Enc: encode(observation)
        Note right of Enc: every sensory neuron gets<br/>0.45 + 0.90*dist/100 + 0.25*vel/8
        Enc-->>Ctl: stimulus (135 values)
        Ctl->>Net: step(stimulus)
        loop 2 substeps
            Note right of Net: drive = 3.0*(W @ state) + stimulus<br/>state += 0.35*(tanh(drive) - state)
        end
        Net-->>Ctl: activity (135 values)
        Ctl->>Dec: decode(activity)
        Note right of Dec: diff = mean(extensor) - mean(flexor)<br/>ACCELERATE if diff > threshold else COAST
        Dec-->>Ctl: action, {extensor, flexor, difference}
        Ctl->>Ctl: store last_summary
        Ctl-->>Run: action
        Run->>Ev: step(action)
        Note right of Ev: v += dt*(force - 0.16*v)<br/>x += dt*v
        Run->>Run: append record (+ last_summary, + neuron_activity if record_activity)
    end
```

## 3. Activity diagram: training and using the trained policy

Training only searches for `motor_threshold`. The connectome weights, leak and
gain never change.

```mermaid
flowchart TD
    A([python -m experiments.train_sprint --seed 7]) --> B["Build 34 thresholds:<br/>0.015, 32 log-spaced values from 0.1 to 1e-8, 0.0"]
    B --> C{Next threshold?}
    C -- yes --> D["FlyBrainController(threshold)"]
    D --> E["run(seed, controller=...)<br/>(full sprint, as in diagram 2)"]
    E --> F{"Finished 100 m?"}
    F -- yes --> G[score = finish time]
    F -- no --> H[score = 60 + distance left]
    G --> I[Save trial result]
    H --> I
    I --> C
    C -- no --> J["Pick the trial with the lowest score"]
    J --> K[("models/sprint.json<br/>version 1, event sprint,<br/>motor_threshold, best, untrained, trials")]

    K -.-> L([python -m experiments.run_sprint --controller trained])
    L --> M["FlyBrainController.trained()<br/>checks the file exists, version == 1, event == sprint"]
    M --> N["FlyBrainController(motor_threshold)"]
    N --> O["run() then write results/*.json, *.csv, *.png<br/>(--view / --gif for animation)"]

    K -.-> P([python -m experiments.serve_view])
    P --> Q["viewer_payload(): circuit nodes/edges +<br/>runs for trained, fly, rule, random<br/>with record_activity=True"]
    Q --> R["HTTP server on 127.0.0.1:8000<br/>serves visualization/web and /runs.json"]
```

## 4. Test map

Run the Python tests with `python -m pytest -q`. The one JavaScript test runs
separately with `node --test tests/test_signals.cjs`.

```mermaid
flowchart LR
    subgraph PY["python -m pytest -q"]
        TS[test_sprint.py]
        TB[test_brain.py]
        TE[test_experiment.py]
        TT[test_training.py]
        TV[test_viewer.py]
    end
    subgraph JS["node --test"]
        TJ[test_signals.cjs]
    end

    TS -->|"reset, determinism,<br/>termination, illegal action"| Sprint
    TB -->|"135 neurons, 2770 edges,<br/>24 sensory"| Conn[connectome.py]
    TB -->|"shapes, finite values,<br/>legal action"| Pipe["SensoryEncoder → RateNetwork → MotorDecoder"]
    TE -->|"random is reproducible;<br/>random, rule, fly all run"| RunS[run_sprint.run]
    TT -->|"train(7) == saved sprint.json;<br/>trained finishes, untrained doesn't"| Train[train_sprint.train]
    TT -->|"trained run matches always-ACCELERATE;<br/>bad thresholds and policy files raise"| FBC[FlyBrainController]
    TV -->|"payload matches circuit;<br/>activity in 0..1; summaries and actions match"| View[serve_view.viewer_payload]
    TJ -->|"signal packets animate;<br/>scrubbing restores position"| Sig[visualization/web/signals.js]

    Train --> RunS
    View --> RunS
    RunS --> FBC
    RunS --> Sprint
    FBC --> Pipe
    FBC --> Conn
    Train -.->|"compared against"| Policy[(models/sprint.json)]
```

**Note:** `test_training.py` checks that `train(seed=7)` reproduces
`models/sprint.json` exactly. Any change to `brain/` or `events/sprint.py` that
changes behavior means you must run `python -m experiments.train_sprint` again
and commit the new policy file. Otherwise that test fails.
