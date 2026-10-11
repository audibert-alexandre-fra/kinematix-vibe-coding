"""Modèle humain MuJoCo paramétrique, généré par code (MjSpec) — pas de dépendance à un
modèle sous licence restrictive.

Repère : Z vers le haut, X vers l'avant du sujet, Y vers sa gauche ; pose neutre debout.
Les longueurs des segments (`BodyDimensions`) sont les paramètres de l'étalonnage
géométrique ; `apply_dimensions` les écrit directement dans `MjModel.body_pos` /
`MjModel.site_pos` sans recompilation (utilisé par la calibration automatique).

Conventions de signe (angles positifs) : flexion de hanche/épaule/coude/tronc/cou,
flexion de genou, dorsiflexion de cheville, adduction de hanche, rotation interne de hanche,
inversion subtalaire, inclinaison du tronc vers la droite, rotation du tronc vers la gauche.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import mujoco
import numpy as np

SIDES = {"r": ("right", -1.0), "l": ("left", 1.0)}


@dataclass
class BodyDimensions:
    """Longueurs des segments (m). Valeurs par défaut : proportions anthropométriques (Winter)."""

    ankle_height: float
    shank: float
    thigh: float
    hip_width: float
    pelvis_to_lumbar: float
    torso: float
    shoulder_width: float
    shoulder_drop: float
    neck_to_eyes: float
    eyes_to_top: float
    upper_arm: float
    forearm: float
    hand: float
    foot_length: float
    heel_back: float

    @classmethod
    def from_height(cls, height: float) -> BodyDimensions:
        h = height
        return cls(
            ankle_height=0.039 * h, shank=0.246 * h, thigh=0.230 * h, hip_width=0.100 * h,
            pelvis_to_lumbar=0.060 * h, torso=0.243 * h, shoulder_width=0.200 * h, shoulder_drop=0.020 * h,
            neck_to_eyes=0.118 * h, eyes_to_top=0.064 * h, upper_arm=0.186 * h, forearm=0.146 * h,
            hand=0.055 * h, foot_length=0.152 * h, heel_back=0.040 * h,
        )

    def standing_height(self) -> float:
        """Hauteur debout (sol -> sommet du crâne) en pose neutre."""
        return (self.ankle_height + self.shank + self.thigh + self.pelvis_to_lumbar + self.torso
                + self.neck_to_eyes + self.eyes_to_top)

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> BodyDimensions:
        return cls(*[float(v) for v in arr])

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ----------------------------------------------------------------------------------------
# Disposition : positions des corps et des sites en fonction des dimensions.
# ----------------------------------------------------------------------------------------

def body_layout(d: BodyDimensions) -> dict[str, np.ndarray]:
    """Position de chaque corps dans le repère de son parent."""
    pos: dict[str, np.ndarray] = {
        "pelvis": np.array([0.0, 0.0, d.ankle_height + d.shank + d.thigh]),
        "torso": np.array([0.0, 0.0, d.pelvis_to_lumbar]),
        "head": np.array([0.0, 0.0, d.torso]),
    }
    for s, (_, sg) in SIDES.items():
        pos[f"thigh_{s}"] = np.array([0.0, sg * d.hip_width / 2, 0.0])
        pos[f"shank_{s}"] = np.array([0.0, 0.0, -d.thigh])
        pos[f"foot_{s}"] = np.array([0.0, 0.0, -d.shank])
        pos[f"clavicle_{s}"] = np.array([0.0, 0.0, d.torso])
        pos[f"upperarm_{s}"] = np.array([0.0, sg * d.shoulder_width / 2, -d.shoulder_drop])
        pos[f"forearm_{s}"] = np.array([0.0, 0.0, -d.upper_arm])
        pos[f"hand_{s}"] = np.array([0.0, 0.0, -d.forearm])
    return pos


def site_layout(d: BodyDimensions) -> dict[str, tuple[str, np.ndarray]]:
    """Sites marqueurs (nom MHR70 -> (corps, position par défaut))."""
    sites: dict[str, tuple[str, np.ndarray]] = {
        "neck": ("head", np.zeros(3)),
        "nose": ("head", np.array([0.09, 0.0, d.neck_to_eyes - 0.03])),
        "head_top": ("head", np.array([0.0, 0.0, d.neck_to_eyes + d.eyes_to_top])),
        "pelvis_center": ("pelvis", np.zeros(3)),
    }
    for s, (side, sg) in SIDES.items():
        sites[f"{side}_eye"] = ("head", np.array([0.07, sg * 0.032, d.neck_to_eyes]))
        sites[f"{side}_ear"] = ("head", np.array([-0.01, sg * 0.075, d.neck_to_eyes - 0.01]))
        sites[f"{side}_hip"] = (f"thigh_{s}", np.zeros(3))
        sites[f"{side}_knee"] = (f"shank_{s}", np.zeros(3))
        sites[f"{side}_ankle"] = (f"foot_{s}", np.zeros(3))
        sites[f"{side}_heel"] = (f"foot_{s}", np.array([-d.heel_back, 0.0, -d.ankle_height + 0.02]))
        toe_x = d.foot_length - d.heel_back
        sites[f"{side}_big_toe_tip"] = (f"foot_{s}", np.array([toe_x, -sg * 0.02, -d.ankle_height + 0.015]))
        sites[f"{side}_small_toe_tip"] = (f"foot_{s}", np.array([toe_x * 0.8, sg * 0.04, -d.ankle_height + 0.015]))
        sites[f"{side}_shoulder"] = (f"upperarm_{s}", np.zeros(3))
        sites[f"{side}_acromion"] = (f"clavicle_{s}", np.array([0.0, sg * (d.shoulder_width / 2 + 0.01), 0.02]))
        sites[f"{side}_elbow"] = (f"forearm_{s}", np.zeros(3))
        sites[f"{side}_olecranon"] = (f"forearm_{s}", np.array([-0.03, 0.0, 0.01]))
        sites[f"{side}_cubital_fossa"] = (f"forearm_{s}", np.array([0.025, 0.0, 0.0]))
        sites[f"{side}_wrist"] = (f"hand_{s}", np.zeros(3))
        sites[f"{side}_index_third_joint"] = (f"hand_{s}", np.array([0.01, 0.0, -d.hand]))
        sites[f"{side}_pinky_third_joint"] = (f"hand_{s}", np.array([-0.035, 0.0, -d.hand * 0.9]))
        sites[f"{side}_middle_third_joint"] = (f"hand_{s}", np.array([-0.01, 0.0, -d.hand * 1.02]))
    # en pose neutre les paumes regardent vers le corps : index devant, auriculaire derrière
    return sites


@dataclass(frozen=True)
class JointDef:
    name: str
    axis: tuple[float, float, float]
    range_deg: tuple[float, float]


def _side_joints(s: str) -> dict[str, list[JointDef]]:
    sg = 1.0 if s == "l" else -1.0  # gauche : +1
    return {
        f"thigh_{s}": [
            JointDef(f"hip_flexion_{s}", (0, -1, 0), (-30, 130)),
            JointDef(f"hip_adduction_{s}", (-sg, 0, 0), (-50, 30)),
            JointDef(f"hip_rotation_{s}", (0, 0, -sg), (-45, 45)),
        ],
        f"shank_{s}": [JointDef(f"knee_flexion_{s}", (0, 1, 0), (-5, 155))],
        f"foot_{s}": [
            JointDef(f"ankle_flexion_{s}", (0, -1, 0), (-50, 40)),
            JointDef(f"subtalar_{s}", (-sg, 0, 0), (-30, 30)),
        ],
        f"clavicle_{s}": [
            JointDef(f"clavicle_elevation_{s}", (sg, 0, 0), (-10, 35)),
            JointDef(f"clavicle_protraction_{s}", (0, 0, sg), (-20, 25)),
        ],
        f"forearm_{s}": [
            JointDef(f"elbow_flexion_{s}", (0, -1, 0), (-5, 155)),
            JointDef(f"pronation_{s}", (0, 0, -sg), (-90, 100)),
        ],
        f"hand_{s}": [
            JointDef(f"wrist_flexion_{s}", (-sg, 0, 0), (-75, 75)),
            JointDef(f"wrist_deviation_{s}", (0, -1, 0), (-35, 35)),
        ],
    }


TRUNK_JOINTS = {
    "torso": [
        JointDef("lumbar_flexion", (0, 1, 0), (-35, 95)),
        JointDef("lumbar_bending", (1, 0, 0), (-40, 40)),
        JointDef("lumbar_rotation", (0, 0, 1), (-45, 45)),
    ],
    "head": [
        JointDef("neck_flexion", (0, 1, 0), (-60, 70)),
        JointDef("neck_bending", (1, 0, 0), (-45, 45)),
        JointDef("neck_rotation", (0, 0, 1), (-80, 80)),
    ],
}

PARENT = {
    "pelvis": "world", "torso": "pelvis", "head": "torso",
    **{f"thigh_{s}": "pelvis" for s in SIDES}, **{f"shank_{s}": f"thigh_{s}" for s in SIDES},
    **{f"foot_{s}": f"shank_{s}" for s in SIDES}, **{f"clavicle_{s}": "torso" for s in SIDES},
    **{f"upperarm_{s}": f"clavicle_{s}" for s in SIDES}, **{f"forearm_{s}": f"upperarm_{s}" for s in SIDES},
    **{f"hand_{s}": f"forearm_{s}" for s in SIDES},
}


def all_joint_defs() -> dict[str, list[JointDef]]:
    out = dict(TRUNK_JOINTS)
    for s in SIDES:
        out.update(_side_joints(s))
    return out


def build_spec(d: BodyDimensions, floor: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "kinematix_humanoid"
    spec.option.gravity = [0, 0, -9.81]
    spec.compiler.degree = False
    if floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
                                rgba=[0.8, 0.8, 0.8, 1], contype=0, conaffinity=0)
    bpos = body_layout(d)
    jdefs = all_joint_defs()
    bodies: dict[str, mujoco.MjsBody] = {}
    order = ["pelvis", "torso", "head"] + [f"{p}_{s}" for s in SIDES for p in
                                           ("thigh", "shank", "foot", "clavicle", "upperarm", "forearm", "hand")]
    for name in order:
        parent = spec.worldbody if PARENT[name] == "world" else bodies[PARENT[name]]
        b = parent.add_body(name=name, pos=bpos[name].tolist())
        bodies[name] = b
        if name == "pelvis":
            b.add_freejoint(name="root")
        elif name.startswith("upperarm"):
            b.add_joint(name=f"shoulder_{name[-1]}", type=mujoco.mjtJoint.mjJNT_BALL)
        for jd in jdefs.get(name, []):
            b.add_joint(name=jd.name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(jd.axis),
                        range=np.deg2rad(jd.range_deg).tolist(), limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    _add_geoms(bodies, d)
    for sname, (body, p) in site_layout(d).items():
        bodies[body].add_site(name=sname, pos=p.tolist(), size=[0.012, 0, 0], rgba=[1, 0.2, 0.2, 1])
    return spec


def _capsule(body, a, b, r, rgba=(0.55, 0.7, 0.95, 1)):
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(a) + list(b), size=[r, 0, 0], rgba=list(rgba),
                  contype=0, conaffinity=0, mass=1.0)


def _add_geoms(bodies, d: BodyDimensions) -> None:
    hw = d.hip_width / 2
    _capsule(bodies["pelvis"], (0, -hw, 0), (0, hw, 0), 0.06)
    _capsule(bodies["torso"], (0, 0, 0), (0, 0, d.torso), 0.09)
    bodies["head"].add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0.02, 0, d.neck_to_eyes], size=[0.1, 0, 0],
                            rgba=[0.95, 0.8, 0.7, 1], contype=0, conaffinity=0, mass=1.0)
    for s, (_, sg) in SIDES.items():
        _capsule(bodies[f"thigh_{s}"], (0, 0, 0), (0, 0, -d.thigh), 0.06)
        _capsule(bodies[f"shank_{s}"], (0, 0, 0), (0, 0, -d.shank), 0.045)
        bodies[f"foot_{s}"].add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[d.foot_length / 2, 0.045, d.ankle_height / 2],
            pos=[d.foot_length / 2 - d.heel_back, 0, -d.ankle_height / 2], rgba=[0.3, 0.3, 0.35, 1],
            contype=0, conaffinity=0, mass=1.0)
        _capsule(bodies[f"clavicle_{s}"], (0, 0, 0), (0, sg * d.shoulder_width / 2, -d.shoulder_drop), 0.03)
        _capsule(bodies[f"upperarm_{s}"], (0, 0, 0), (0, 0, -d.upper_arm), 0.04)
        _capsule(bodies[f"forearm_{s}"], (0, 0, 0), (0, 0, -d.forearm), 0.035)
        _capsule(bodies[f"hand_{s}"], (0, 0, 0), (0, 0, -d.hand), 0.03)


class Humanoid:
    """MjModel + MjData du modèle humain, avec accès par nom aux marqueurs et articulations."""

    def __init__(self, dims: BodyDimensions):
        self.dims = dims
        self.spec = build_spec(dims)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self.site_offsets = np.zeros((self.model.nsite, 3))  # corrections de calibration
        self.hinge_names = [self.model.joint(i).name for i in range(self.model.njnt)
                            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.reset()

    def reset(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        mujoco.mj_kinematics(self.model, self.data)

    def site_id(self, name: str) -> int:
        return self.site_names.index(name)

    def apply_dimensions(self, dims: BodyDimensions, site_offsets: np.ndarray | None = None) -> None:
        """Met à jour les longueurs de segments (et offsets des marqueurs) dans le MjModel compilé."""
        self.dims = dims
        if site_offsets is not None:
            self.site_offsets = np.asarray(site_offsets, float).copy()
        for name, p in body_layout(dims).items():
            self.model.body_pos[self.model.body(name).id] = p
        for sname, (_, p) in site_layout(dims).items():
            i = self.site_id(sname)
            self.model.site_pos[i] = p + self.site_offsets[i]

    def rebuilt(self) -> Humanoid:
        """Nouveau modèle compilé (géométries de rendu à l'échelle) avec les dimensions courantes."""
        h = Humanoid(self.dims)
        h.apply_dimensions(self.dims, self.site_offsets)
        return h

    def joint_qposadr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self.model.joint(name).id])

    def hinge_angles(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Angles (rad) des articulations pivots pour une trajectoire qpos (T, nq)."""
        qpos = np.atleast_2d(qpos)
        return {n: qpos[:, self.joint_qposadr(n)] for n in self.hinge_names}

    def sites_world(self, qpos: np.ndarray) -> np.ndarray:
        """Positions monde (T, nsite, 3) des marqueurs pour une trajectoire qpos."""
        qpos = np.atleast_2d(qpos)
        out = np.zeros((len(qpos), self.model.nsite, 3))
        for t, q in enumerate(qpos):
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[t] = self.data.site_xpos
        return out

    def neutral_height(self) -> float:
        """Hauteur debout mesurée dans MuJoCo (sommet du crâne - sol) en pose neutre."""
        q = self.model.qpos0.copy()
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        top = self.data.site_xpos[self.site_id("head_top")][2]
        sole = self.data.xpos[self.model.body("foot_l").id][2] - self.dims.ankle_height
        return float(top - sole)
