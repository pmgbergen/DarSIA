"""Identity flash: the signal is the aqueous concentration, without clipping."""

import numpy as np

import darsia


def test_signal_is_the_concentration_as_it_is() -> None:
    signal = darsia.Image(np.array([[-0.1, 0.5], [1.0, 1.2]]), width=1.0, height=1.0)
    c_aq, s_g = darsia.IdentityFlash()(signal)
    assert np.array_equal(c_aq.img, signal.img)
    assert c_aq is not signal
    assert np.array_equal(s_g.img, np.zeros_like(signal.img))
