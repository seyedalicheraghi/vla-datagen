"""Checks of the RTX look presets: well formed, applied through the setter, ambient fill off (pure Python).

    python scripts/drone_scene/test/test_rtx_look.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drone_scene import rtx_look as R  # noqa: E402

AMBIENT = "/rtx/sceneDb/ambientLightIntensity"


class FakeKit:
    """A settings store with the sim.set_setting / sim.get_setting interface; ``registered`` limits get_setting."""

    def __init__(self, registered=None, values=None):
        self.registered = registered
        self.values = dict(values or {})
        self.calls: list[tuple[str, object]] = []

    def set_setting(self, key, value):
        self.calls.append((key, value))
        self.values[key] = value

    def get_setting(self, key):
        if self.registered is not None and key not in self.registered and key not in self.values:
            return None
        return self.values.get(key, 0)


class TestPresets(unittest.TestCase):
    def test_presets_are_well_formed(self):
        self.assertEqual(set(R.SETTINGS), {"realtime", "realtime2", "pathtracing"})
        groups = list(R.SETTINGS.items()) + list(R.OPTIONAL.items())
        for name, preset in groups:
            self.assertTrue(preset, name)
            for key, value in preset.items():
                self.assertIsInstance(key, str)
                self.assertRegex(key, r"^/rtx(-transient)?/[A-Za-z0-9_]+(/[A-Za-z0-9_]+)*$", f"{name}: {key}")
                self.assertIsInstance(value, (bool, int, float, str), f"{name}: {key}")
                if isinstance(value, float):
                    self.assertTrue(value == value and abs(value) != float("inf"), f"{name}: {key}")

    def test_every_key_has_provenance(self):
        tags = {"source", "docs", "repo", "example", "unverified"}
        for preset in list(R.SETTINGS.values()) + list(R.OPTIONAL.values()):
            for key in preset:
                self.assertIn(R.PROVENANCE.get(key), tags, key)
        for key, tag in R.PROVENANCE.items():
            self.assertIn(tag, tags, key)
        self.assertEqual(R.unverified_keys(), sorted(R.unverified_keys()))
        self.assertIn("/rtx/post/dof/enabled", R.unverified_keys())

    def test_ambient_fill_is_off_in_every_preset(self):
        for name, preset in R.SETTINGS.items():
            self.assertEqual(preset[AMBIENT], 0.0, name)
            self.assertIsInstance(preset[AMBIENT], float, name)

    def test_modes_and_shared_look(self):
        self.assertEqual(R.SETTINGS["pathtracing"]["/rtx/rendermode"], "PathTracing")
        self.assertEqual(R.SETTINGS["realtime"]["/rtx/rendermode"], "RaytracedLighting")
        for name, preset in R.SETTINGS.items():
            self.assertIs(preset["/rtx/post/histogram/enabled"], False, name)    # fixed exposure
            self.assertNotIn("/rtx/post/tonemap/op", preset, name)              # enum unsettled: opt-in
            self.assertNotIn("/rtx/post/aa/op", preset, name)                   # Isaac Lab sets DLAA
        self.assertEqual(R.OPTIONAL["tonemap_aces"], {"/rtx/post/tonemap/op": 6})
        self.assertEqual(R.SETTINGS["realtime"]["/rtx/domeLight/upperLowerStrategy"], 4)   # sunless sky + sun light
        self.assertNotIn("/rtx/domeLight/upperLowerStrategy", R.SETTINGS["pathtracing"])

    def test_no_exposure_keys_by_default(self):
        for name, preset in R.SETTINGS.items():
            self.assertFalse([k for k in preset if "/tonemap/" in k and not k.endswith("/op")], name)

    def test_pathtracing_preset_mirrors_simulation_app(self):
        # Isaac Sim 5.1's SimulationApp sets spp, totalSpp and clampSpp to one number (default 64): one launch
        # per frame, and no assumed 32-sample cap
        pt = R.SETTINGS["pathtracing"]
        spp = pt["/rtx/pathtracing/spp"]
        self.assertEqual(pt["/rtx/pathtracing/totalSpp"], spp)
        self.assertEqual(pt["/rtx/pathtracing/clampSpp"], spp)
        self.assertEqual(spp, 64)
        for key in R.PT_SAMPLE_KEYS[:3]:
            self.assertEqual(R.PROVENANCE[key], "source", key)
        self.assertIn("/rtx/pathtracing/clampSpp", R.UNVERIFIED_VALUES)
        self.assertIn("/rtx/pathtracing/maxSamplesPerLaunch", R.PT_SAMPLE_KEYS)
        self.assertNotIn("/rtx/pathtracing/clampSpp", R.SETTINGS["realtime"])


class TestPathTracingSamples(unittest.TestCase):
    def test_default_is_one_launch_with_all_samples(self):
        # the integration regression: --spp 64 must stay 64 samples in one render call, not 32 + a second launch
        for total in (1, 16, 32, 33, 64, 100, 128, 1000):
            got, n = R.path_tracing_samples(total)
            self.assertEqual(n, 1, total)
            self.assertEqual(got, {"/rtx/pathtracing/spp": total, "/rtx/pathtracing/totalSpp": total,
                                   "/rtx/pathtracing/clampSpp": total}, total)

    def test_default_matches_the_preset_keys(self):
        got, _ = R.path_tracing_samples(64)
        self.assertEqual(got, {k: R.SETTINGS["pathtracing"][k] for k in got})

    def test_opt_in_launch_split(self):
        for total, spp, launches in ((1, 1, 1), (16, 16, 1), (32, 32, 1), (33, 17, 2), (64, 32, 2), (100, 25, 4)):
            got, n = R.path_tracing_samples(total, R.PT_MAX_SPP_PER_LAUNCH)
            self.assertEqual((got["/rtx/pathtracing/spp"], n), (spp, launches), total)
            self.assertEqual(got["/rtx/pathtracing/totalSpp"], spp * launches, total)
            self.assertEqual(got["/rtx/pathtracing/clampSpp"], spp * launches, total)

    def test_split_never_exceeds_a_launch_and_never_short(self):
        for limit in (1, 8, 32):
            for total in range(1, 400):
                got, n = R.path_tracing_samples(total, limit)
                self.assertLessEqual(got["/rtx/pathtracing/spp"], limit)
                self.assertGreaterEqual(got["/rtx/pathtracing/totalSpp"], total)
                self.assertLess(got["/rtx/pathtracing/totalSpp"] - total, n)     # rounding adds < 1 sample per launch
                self.assertEqual(got["/rtx/pathtracing/clampSpp"], got["/rtx/pathtracing/totalSpp"])

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            R.path_tracing_samples(0)
        with self.assertRaises(ValueError):
            R.path_tracing_samples(64, 0)


class TestApply(unittest.TestCase):
    def run_apply(self, preset, kit=None, **kw):
        kit = kit or FakeKit()
        logs: list[str] = []
        applied = R.apply(kit.set_setting, kit.get_setting, preset, logs.append, **kw)
        return kit, applied, logs

    def test_setter_called_for_every_key(self):
        for preset in R.SETTINGS:
            kit, applied, logs = self.run_apply(preset)
            self.assertEqual(applied, R.SETTINGS[preset])
            self.assertEqual(dict(kit.calls), R.SETTINGS[preset])
            self.assertEqual(len(kit.calls), len(R.SETTINGS[preset]))
            self.assertFalse([m for m in logs if "WARNING" in m], preset)

    def test_warns_for_unregistered_keys_but_still_sets_them(self):
        missing = {AMBIENT, "/rtx/indirectDiffuse/maxBounces"}
        registered = set(R.SETTINGS["realtime"]) - missing
        kit, applied, logs = self.run_apply("realtime", FakeKit(registered=registered))
        warned = [m for m in logs if "WARNING" in m]
        self.assertEqual(len(warned), 2)
        for key in missing:
            self.assertEqual(sum(key in m for m in warned), 1, key)
            self.assertIn(key, applied)
            self.assertIn((key, R.SETTINGS["realtime"][key]), kit.calls)

    def test_nothing_registered_does_not_fail(self):
        kit, applied, logs = self.run_apply("pathtracing", FakeKit(registered=set()))
        self.assertEqual(applied, R.SETTINGS["pathtracing"])
        self.assertEqual(sum("WARNING" in m for m in logs), len(R.SETTINGS["pathtracing"]))

    def test_unknown_preset_and_group(self):
        with self.assertRaises(ValueError):
            self.run_apply("raytraced")
        kit = FakeKit()
        with self.assertRaises(ValueError):
            self.run_apply("realtime", kit, optional=("bloom",))
        self.assertEqual(kit.calls, [])                       # validated before anything is set

    def test_optional_groups_and_overrides(self):
        _, applied, _ = self.run_apply("realtime", optional=("lens", "motion_blur"))
        for group in ("lens", "motion_blur"):
            for key, value in R.OPTIONAL[group].items():
                self.assertEqual(applied[key], value, key)
        for key in ("exposureFraction", "numSamples", "maxBlurDiameterFraction"):     # NVIDIA's real-time example
            self.assertIn(f"/rtx/post/motionblur/{key}", R.OPTIONAL["motion_blur"])
        _, applied, _ = self.run_apply("realtime", optional="dof")            # a bare string is one group
        self.assertTrue(applied["/rtx/post/dof/enabled"])
        _, applied, _ = self.run_apply("pathtracing", overrides={"/rtx/pathtracing/spp": 7, AMBIENT: 0.5})
        self.assertEqual(applied["/rtx/pathtracing/spp"], 7)
        self.assertEqual(applied[AMBIENT], 0.5)
        self.assertEqual(R.SETTINGS["pathtracing"]["/rtx/pathtracing/spp"], 64)   # presets are not mutated

    def test_path_tracing_samples_as_overrides(self):
        settings, launches = R.path_tracing_samples(128)
        kit, applied, _ = self.run_apply("pathtracing", overrides=settings)
        sent = dict(kit.calls)
        self.assertEqual(launches, 1)
        for key in ("spp", "totalSpp", "clampSpp"):
            self.assertEqual((applied[f"/rtx/pathtracing/{key}"], sent[f"/rtx/pathtracing/{key}"]), (128, 128), key)
        self.assertEqual(len(kit.calls), len(R.SETTINGS["pathtracing"]))          # overrides replace, not add

    def test_split_samples_as_overrides(self):
        settings, launches = R.path_tracing_samples(64, R.PT_MAX_SPP_PER_LAUNCH)
        kit, applied, _ = self.run_apply("pathtracing", overrides=settings)
        self.assertEqual((applied["/rtx/pathtracing/spp"], applied["/rtx/pathtracing/totalSpp"],
                          applied["/rtx/pathtracing/clampSpp"], launches), (32, 64, 64, 2))

    def test_pathtracing_logs_sample_limits_before_setting_them(self):
        kit = FakeKit(values={"/rtx/pathtracing/clampSpp": 8, "/rtx/pathtracing/maxSamplesPerLaunch": 1000000})
        _, _, logs = self.run_apply("pathtracing", kit)
        found = [m for m in logs if "sample limits before" in m]
        self.assertEqual(len(found), 1)
        self.assertIn("clampSpp=8", found[0])
        self.assertIn("maxSamplesPerLaunch=1000000", found[0])
        self.assertFalse([m for m in logs if "WARNING" in m])                     # information, not a warning
        self.assertEqual(kit.values["/rtx/pathtracing/clampSpp"], 64)
        _, _, logs = self.run_apply("realtime")
        self.assertFalse([m for m in logs if "sample limits" in m])

    def test_exposure_ev_scales_film_iso(self):
        kit = FakeKit(values={R.FILM_ISO: 100.0})
        _, applied, logs = self.run_apply("realtime", kit, exposure_ev=1.0)
        self.assertEqual(applied[R.FILM_ISO], 200.0)
        self.assertEqual(kit.values[R.FILM_ISO], 200.0)
        self.assertFalse([m for m in logs if "WARNING" in m])
        kit = FakeKit(values={R.FILM_ISO: 400})
        self.run_apply("realtime", kit, exposure_ev=-2)
        self.assertEqual(kit.values[R.FILM_ISO], 100.0)

    def test_exposure_untouched_by_default_and_when_unreadable(self):
        kit, applied, _ = self.run_apply("realtime", FakeKit(values={R.FILM_ISO: 100.0}))
        self.assertNotIn(R.FILM_ISO, applied)
        self.assertEqual(kit.values[R.FILM_ISO], 100.0)
        kit, applied, logs = self.run_apply("realtime", FakeKit(registered=set()), exposure_ev=1.0)
        self.assertNotIn(R.FILM_ISO, applied)
        self.assertNotIn(R.FILM_ISO, [k for k, _ in kit.calls])
        self.assertTrue([m for m in logs if "exposure_ev" in m])


if __name__ == "__main__":
    unittest.main()
