import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from hand_motion_tracker import (
    HandState,
    ShutdownCountdown,
    angle_degrees,
    correct_handedness,
    parse_args,
    reset_motion,
    run_shutdown_command,
    update_motion,
)


class HandMotionTrackerTests(unittest.TestCase):
    def test_reset_motion_prevents_old_velocity_from_affecting_new_tracking(self):
        state = HandState(
            previous_center=(100.0, 100.0),
            previous_time=1.0,
            smoothed_velocity=(500.0, -300.0),
        )

        reset_motion(state)
        velocity = update_motion(state, (200.0, 200.0), 3.0)

        self.assertEqual(velocity, (0.0, 0.0, 0.0, "diam"))
        self.assertEqual(state.smoothed_velocity, (0.0, 0.0))
        self.assertEqual(state.previous_center, (200.0, 200.0))

    def test_angle_degrees_returns_straight_angle(self):
        self.assertAlmostEqual(angle_degrees((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), 180.0)

    def test_handedness_is_corrected_for_unmirrored_input(self):
        self.assertEqual(correct_handedness("Left", mirrored_input=False), "Right")
        self.assertEqual(correct_handedness("Right", mirrored_input=False), "Left")
        self.assertEqual(correct_handedness("Left", mirrored_input=True), "Left")

    def test_video_and_headless_options_are_parsed(self):
        args = parse_args(["--video", "sample.mp4", "--no-display", "--no-csv"])

        self.assertEqual(str(args.video), "sample.mp4")
        self.assertTrue(args.no_display)
        self.assertTrue(args.no_csv)

    def test_virtual_camera_option_is_parsed(self):
        args = parse_args(["--virtual-camera", "--no-csv"])

        self.assertTrue(args.virtual_camera)
        self.assertEqual(args.virtual_camera_backend, "obs")
        self.assertTrue(args.no_csv)

    def test_unity_capture_backend_is_parsed(self):
        args = parse_args(["--virtual-camera", "--virtual-camera-backend", "unitycapture"])

        self.assertTrue(args.virtual_camera)
        self.assertEqual(args.virtual_camera_backend, "unitycapture")

    def test_virtual_camera_rejects_video_files(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--video", "sample.mp4", "--virtual-camera"])

    def test_invalid_dimensions_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--width", "0"])

    @patch("hand_motion_tracker.subprocess.run")
    @patch.dict("os.environ", {"SystemRoot": r"C:\Windows"})
    def test_shutdown_is_scheduled_through_windows_shutdown_exe(self, run):
        run_shutdown_command("/s", "/t", "15", "/f")

        run.assert_called_once_with(
            [str(Path(r"C:\Windows") / "System32" / "shutdown.exe"), "/s", "/t", "15", "/f"],
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("hand_motion_tracker.run_shutdown_command")
    def test_countdown_submits_shutdown_once_only_after_deadline(self, shutdown_command):
        countdown = ShutdownCountdown(deadline=10.0)

        self.assertFalse(countdown.submit_if_due(9.99))
        shutdown_command.assert_not_called()
        self.assertTrue(countdown.submit_if_due(10.0))
        shutdown_command.assert_called_once_with("/s", "/t", "0", "/f")
        self.assertFalse(countdown.submit_if_due(11.0))
        self.assertTrue(countdown.command_sent)

    @patch("hand_motion_tracker.run_shutdown_command", side_effect=RuntimeError("failed"))
    def test_countdown_reports_shutdown_failure_and_requires_pose_release(self, shutdown_command):
        countdown = ShutdownCountdown(deadline=10.0)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            countdown.submit_if_due(10.0)

        shutdown_command.assert_called_once_with("/s", "/t", "0", "/f")
        self.assertIsNone(countdown.deadline)
        self.assertTrue(countdown.require_pose_release)
        self.assertFalse(countdown.command_sent)


if __name__ == "__main__":
    unittest.main()
