# POC validation scenarios

The validation runner turns recorded scenarios into repeatable pass/fail checks.
Copy `manifest.example.json`, place consented test media in this folder, and
adjust the expectations for the scene.

Run:

```powershell
.\.venv\Scripts\python.exe src\main.py --validate --manifest validation\manifest.json
```

Recommended minimum scenario set:

1. One enrolled person in normal indoor lighting.
2. One unknown person who must remain unknown.
3. Two or more people crossing paths and briefly occluding each other.
4. A person leaving and re-entering after their tracker expires.
5. No face visible while pose tracking continues.
6. Repeated right-hand waves followed by enough idle time for the tier to recover.
7. Low light, side profile, glasses, and increased camera distance.

Before calling the POC demo-ready, require all functional cases to pass on
Windows, macOS, and Linux, and record the measured processing FPS for each
target machine. Do not commit real enrollment or validation footage without
the participants' permission.
