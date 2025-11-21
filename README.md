# Gear Train Assembly
We will train two skills for the manipulator
- Peg Insertion
- Gear Meshing

## Requirements
- Isaac Lab [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html]

## Setup
1. Go to your IsaacLab directory `source/isaaclab_tasks/isaaclab_tasks/manager_based`
2. Clone the repository `https://github.com/bugartlan/HCR_Gear_Train_Assembly.git`

## Hardware
- UR3e
- Robotiq Hand-E

### Training 
- **Train on HCRL servers**: Same command as how you run SO_101, except replacing the task name with
```bash
./ray.sh job --task Isaac-Assembly-PegInsert-Chamfer-v0
```

- **Run training without visualization (headless)**: Use `--num_env 1` for faster startup for quick syntax checks.
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Assembly-PegInsert-v0 --headless
```
- **Run training with visualization** for visual debugging.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Assembly-PegInsert-v0 --num_env 1
```

### Playing a Policy
- **Export and run the most recent trained policy**:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Assembly-PegInsert-v0 --num_env 1
```

