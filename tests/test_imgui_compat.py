"""The backend reordered rectangle thickness/flags in newer ImGui releases."""

from utils.imgui_compat import _DrawListWrapper


def test_rect_parameters_survive_backend_reordering():
    calls = []

    class Backend:
        def add_rect(self, p_min, p_max, col, rounding=0.0, thickness=1.0, flags=0):
            calls.append((p_min.x, p_max.y, col, rounding, thickness, flags))

    _DrawListWrapper(Backend()).add_rect(
        1, 2, 3, 4, 123, rounding=2, flags=16, thickness=3.5
    )
    assert calls == [(1, 4, 123, 2, 3.5, 16)]
