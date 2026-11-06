# Fruit fly body meshes

These files come from FlyGym's NeuroMechFly model:
https://github.com/NeLy-EPFL/flygym (commit in `SOURCE_COMMIT`), licensed under Apache-2.0
(see `LICENSE-flygym`).

- `meshes/`: the simplified (at most 2,000 faces) left-side and centre body parts, from
  `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/`. Right-side parts
  are mirrored from the left ones when the viewer loads them. Units are metres.
- `rigging.yaml`: where each part attaches to its parent, in millimetres.
- `neutral_pose.yaml`: resting joint angles, from `pose/neutral/yaw_pitch_roll.yaml`.
- `visuals.yaml`: FlyGym's colours, used as the reference for the viewer's materials.

The meshes come from a micro-CT scan of an adult female *Drosophila melanogaster*:

Lobato-Rios, V., et al. (2022). NeuroMechFly, a neuromechanical model of adult
*Drosophila melanogaster*. *Nature Methods*.

The files are unmodified copies. The walking gait and wingbeat in `fly3d.js` are Flympics
animation, not part of FlyGym.
