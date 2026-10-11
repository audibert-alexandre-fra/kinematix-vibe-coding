import importlib


def test_package_imports():
    for mod in ["", ".models", ".calibration", ".inverse_kinematix", ".video", ".utils"]:
        importlib.import_module("kinematix_vibe_coding" + mod)


def test_mujoco_available():
    import mujoco

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq == 7
