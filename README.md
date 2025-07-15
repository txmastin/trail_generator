# Trail Generator

A trail generating script for creating trails in the style of the "Santa Fe Trail Problem," a common reinforcement learning and genetic programming benchmark.

## Background

This script provides a graphical tool for generating 2D paths based on the movement of a probabilistic agent. The concept is inspired by classic agent-based modeling problems like the "Santa Fe Trail."

The core of this tool is a "trail-generating agent" that moves on a grid. At each step, the agent's behavior is governed by two primary probabilities:

* **Tortuosity:** The probability that the agent will attempt to turn left or right, rather than continue straight. Higher values result in more winding, complex paths.
* **Sparsity:** The likelihood that the agent will "forget" to leave a food pellet at its current location, creating gaps in the trail. Higher values result in sparser trails.

The agent is designed with movement constraints to create high-quality, non-trivial trails. For example, it cannot reverse its direction and is "insulated" from its own trail, meaning it cannot move directly adjacent to a previously laid path. This prevents the trail from clumping together and ensures a continuous path.

## How to Use

### Prerequisites

You must have Python installed, along with the `pygame` and `numpy` libraries.

### Running the Tool

1.  Save the Python script provided as a `.py` file (e.g., `trail_generator.py`).
2.  Run the script from your terminal:
    ```
    python trail_generator.py
    ```

### Generating a Trail

1.  Upon launching, you will see the **"Trail Generator Settings"** screen.
2.  Fill in the following fields:
    * **Trail Name:** A name for your trail, which will be used as the filename.
    * **Grid Size:** The width and height of the square grid (e.g., `50`).
    * **Tortuosity (0-1):** The probability of turning. `0.0` will result in a straight line; higher values give a more difficult trail.
    * **Sparsity (0-1):** The probability of leaving a gap. `0.0` creates a fully connected trail; higher values give a more difficult trail.
    * **Trail Length (0=max):** The number of steps the agent should take. If set to `0`, the agent will run until it gets trapped.
3.  Click the **"Generate Trail"** button.

### Saving the Trail

1.  The simulation will run in real-time. When it finishes (either by trapping the agent or reaching the specified length), the simulation will pause.
2.  Click the **"Save Trail"** button.
3.  A `.txt` file will be saved in the same directory as the script. The file will be named according to the "Trail Name" you provided.
4.  This file contains the coordinates of every food pellet in `(x, y)` format, one coordinate per line.
