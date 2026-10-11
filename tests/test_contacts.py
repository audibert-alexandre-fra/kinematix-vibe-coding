import numpy as np

from kinematix_vibe_coding.inverse_kinematix.contacts import ContactConfig, detect_contacts, foot_sliding


def stepping_foot(T=120, fps=30.0, belt=0.0, slide=0.0):
    """Un point de pied : appui 0.5 s (avec glissement parasite `slide` m/s), vol 0.5 s."""
    t = np.arange(T) / fps
    p = np.zeros((T, 1, 3))
    x = 0.0
    for i in range(T):
        stance = (t[i] % 1.0) < 0.5
        if stance:
            x += (belt + slide) / fps
            p[i, 0] = [x, 0, 0.0]
        else:
            x += 1.2 / fps
            p[i, 0] = [x, 0, 0.1 * np.sin(np.pi * ((t[i] % 1.0) - 0.5) / 0.5)]
    return p


def test_detect_contacts_ground():
    p = stepping_foot(slide=0.05)
    c = detect_contacts(p, np.zeros(1), 30.0)
    assert 0.3 < c.mask.mean() < 0.6
    np.testing.assert_allclose(c.belt_velocity, 0)
    # cibles fixes pendant chaque appui
    tgt = c.targets[c.mask[:, 0], 0]
    assert np.isfinite(tgt).all() and np.allclose(tgt[:, 2], 0)
    s = foot_sliding(p, c.mask, c.belt_velocity, 30.0)
    assert s["sliding_speed_mean_cm_s"] > 3  # le glissement parasite est mesuré


def test_detect_contacts_treadmill():
    p = stepping_foot(belt=-1.0)
    c = detect_contacts(p, np.zeros(1), 30.0, ContactConfig())
    assert c.belt_velocity[0] < -0.8
    assert c.mask.mean() > 0.3
    s = foot_sliding(p, c.mask, c.belt_velocity, 30.0)
    assert s["sliding_speed_mean_cm_s"] < 5  # pas de glissement relatif au tapis


def test_overground_walking_is_not_a_treadmill():
    """Marche au sol : les pieds bas bougent (bruit, fin d'appui) mais le bassin avance -> pas de tapis."""
    p = stepping_foot(belt=-0.8)  # mêmes vitesses de pied que sur tapis...
    root = np.zeros((len(p), 3))
    root[:, 0] = np.arange(len(p)) / 30.0 * 1.2  # ...mais le bassin avance à 1,2 m/s
    c = detect_contacts(p, np.zeros(1), 30.0, root_pos=root)
    np.testing.assert_allclose(c.belt_velocity, 0)
