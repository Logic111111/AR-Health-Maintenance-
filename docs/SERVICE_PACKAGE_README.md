# LECO AR Data Generator: Windows service

This package runs the **real-time data generator** for the LECO auto-recloser (AR) health project as a Windows
service.

Every 15 minutes it:
1. collects one interval of data for each NOJA OSM recloser. The data comes from a deterministic, physics-based
   simulator of 100 LECO reclosers.
2. calculates the 42 variables V1–V42;
3. calculates the **Health Index (0–100)**;
4. stores everything in a database: SQLite (built in) or MySQL / MariaDB.

After installation, the service:
- starts by itself whenever Windows starts, without anyone logging in;
- catches up on any time the PC was off;
- is restarted by Windows if it ever crashes.

The dashboard and the machine-learning parts of the project are **not** included.

## What you need

- Windows 10 or 11.
- **Python 3.13 (64-bit)** from https://www.python.org/downloads/. This is the version the package was built and
  tested with.
  - In the installer, tick **"Add python.exe to PATH"**.
  - **"Install for all users"** is recommended.
- An internet connection once, to install the Python packages.
- Administrator rights once, to register the service.

## Install (about 15 minutes, once)

1. **Unzip** the package to a permanent folder, for example `C:\LECO_AR_Generator`. The service runs from this
   folder, so don't delete or move it afterwards.
2. **Open PowerShell as administrator:** Start menu → type `PowerShell` → right-click **Windows PowerShell** →
   **Run as administrator** → **Yes**.
3. **Install the Python packages:**

   ```powershell
   cd "C:\LECO_AR_Generator"
   python -m pip install -r requirements-service.txt
   ```

4. **Optional: use MySQL / MariaDB instead of SQLite.** Copy `.env.example` to `.env` and set `DATABASE_URL` in it.
   Without a `.env` file, the data goes to `data\ar_maintenance.sqlite` in this folder.
5. **Create the database and the recloser fleet with 30 days of history.** The backfill takes about 8 minutes.

   ```powershell
   python main.py init-db
   python main.py seed --days 30
   python main.py backfill
   python main.py validate          # must end with "Validation PASSED"
   ```

6. **Register and start the service:**

   ```powershell
   python main.py service install   # test start, register, start
   python main.py service status    # should say RUNNING, with the last run and the next run time
   ```

That's it. From now on the data stays up to date every 15 minutes.

## Everyday use

| Task | Command (in this folder) |
|---|---|
| Is it running? Last run, data time, errors | `python main.py service status` (normal PowerShell) |
| Stop / start / restart | `python main.py service stop` / `start` / `restart` (admin PowerShell) |
| Latest Health Index of every recloser | `python main.py status` |
| Check the data for consistency | `python main.py validate` |
| Remove the service | `python main.py service remove` (admin PowerShell) |
| Run without the service, in this window | `python main.py service run` (Ctrl+C stops it) |

You can also use the Windows **Services** window: press **Win+R**, type `services.msc`, then find
**LECO AR Data Generator**.

To start it only by hand instead of at every boot (admin PowerShell):
- Manual start: `Set-Service LecoArDataGenerator -StartupType Manual`
- Back to automatic: `sc.exe config LecoArDataGenerator start= delayed-auto`

## Where things are

| What | Where |
|---|---|
| Service log | `logs\ar_service.log`. Windows events: Event Viewer → Windows Logs → Application |
| Database (SQLite) | `data\ar_maintenance.sqlite`, or the `DATABASE_URL` in `.env` |
| Simulator state | `state\sim_state.json`: the fleet's hidden state. **Keep it with the database** |
| Last run of the generator | `state\generator_status.json` |
| All settings | `config.yaml` |
| Every table and column | `DATA_DICTIONARY.md` |

## Good to know

- **One generator per database.** Never point two copies of this service (or this service and another PC) at the
  same database: each one has its own simulator state, and they would write over each other.
- **Moved the folder or reinstalled Python?** Run `python main.py service install` again in an admin PowerShell.
- **"Access is denied" / "needs administrator rights":** the PowerShell is not running as administrator.
- **`service install` says the test start failed:** a Python package is missing. Run step 3 again, then
  `service install`.
- **Same fleet as the sender.** With the default `config.yaml`, `seed --start YYYY-MM-DD` with the same first day
  produces exactly the same reclosers and data, because the simulator is deterministic.
- **This is simulated data.** It is shaped like real NOJA RC-10 / RC-15 controller data, so a real data source can
  replace the simulator later.
