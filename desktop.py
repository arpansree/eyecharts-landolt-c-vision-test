"""Desktop capture, EyeCharts keyboard controls, and spoken prompts."""
import ctypes
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager

import mss
import numpy as np
from PIL import Image

from chart_reader import ChartError


class EyeChartsDesktop:
    def __init__(self, monitor):
        self.system = platform.system()
        self.monitor = monitor
        if self.system not in ("Darwin", "Windows"):
            raise ChartError("Live EyeCharts control is supported on macOS and Windows.")

    @contextmanager
    def session(self):
        if self.system == "Darwin":
            import Quartz

            if not Quartz.CGPreflightScreenCaptureAccess():
                Quartz.CGRequestScreenCaptureAccess()
                raise ChartError("Allow Screen Recording for the launcher in macOS Privacy & Security, then relaunch.")
            if not Quartz.CGPreflightPostEventAccess():
                Quartz.CGRequestPostEventAccess()
                raise ChartError("Allow Accessibility for the launcher in macOS Privacy & Security, then relaunch.")
        with mss.MSS(with_cursor=False) as self.capture:
            if not 1 <= self.monitor < len(self.capture.monitors):
                raise ChartError(f"Display {self.monitor} is unavailable.")
            yield self

    def foreground(self):
        if self.system == "Darwin":
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if app and app.bundleIdentifier() == "org.eyecharts.app":
                return app.processIdentifier()
            return None

        import psutil
        from ctypes import wintypes

        api = ctypes.windll.user32
        api.GetForegroundWindow.restype = wintypes.HWND
        api.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        window = api.GetForegroundWindow()
        pid = wintypes.DWORD()
        if window and api.GetWindowThreadProcessId(window, ctypes.byref(pid)):
            try:
                if psutil.Process(pid.value).name().casefold() == "eyecharts.exe":
                    return pid.value
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def wait_for_focus(self):
        print("Switch to fullscreen EyeCharts on the selected monitor.", flush=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.foreground() is not None:
                time.sleep(0.6)
                return
            time.sleep(0.2)
        raise ChartError("EyeCharts did not come to the foreground.")

    def screenshot(self):
        pid = self.foreground()
        if pid is None:
            raise ChartError("EyeCharts lost focus. Return to the fullscreen chart.")
        shot = self.capture.grab(self.capture.monitors[self.monitor])
        if self.foreground() != pid:
            raise ChartError("The foreground app changed during capture.")
        return np.asarray(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX").convert("L"))

    def step(self, smaller):
        pid = self.foreground()
        if pid is None:
            raise ChartError("EyeCharts lost focus. No advance key was sent.")
        if self.system == "Darwin":
            import Quartz

            key = 125 if smaller else 126
            for pressed in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, key, pressed)
                Quartz.CGEventPostToPid(pid, event)
        else:
            key = 0x28 if smaller else 0x26
            api = ctypes.windll.user32
            api.keybd_event(key, 0, 1, 0)
            api.keybd_event(key, 0, 1 | 2, 0)
        time.sleep(0.45)

    def say(self, message):
        print(message, flush=True)
        if self.system == "Darwin":
            command = ["/usr/bin/say", message]
            options = {}
        else:
            powershell = shutil.which("powershell.exe")
            if not powershell:
                raise ChartError("Windows PowerShell is required for spoken prompts.")
            text = message.replace("'", "''")
            command = [
                powershell, "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speaker.Speak('{text}')",
            ]
            options = {"creationflags": subprocess.CREATE_NO_WINDOW}
        subprocess.run(command, check=True, timeout=30, **options)
