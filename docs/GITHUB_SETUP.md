# Daily auto-run on GitHub (about 10 minutes, once)

Why: free Supabase projects pause after 7 days without activity. This makes GitHub run the
whole pipeline every morning, even when your laptop is off. It is free (private repo,
~5 min/day of 2,000 free min/month).

## 0. First: replace the leaked password and key
Both were pasted in a chat earlier, so make new ones before giving them to GitHub.
- **Supabase**: your project -> Project Settings -> Database -> Reset database password.
  Then put the new password into your `.env` (the `DATABASE_URL` line).
- **Gemini**: aistudio.google.com -> API keys -> delete the old key -> Create API key.
  Then run `scripts\set_gemini_key.bat` with the new key.
- Run `scripts\run_all.bat` (or `python -m dashboard.db`) to check the connection still works.

## 1. Put the project on GitHub (private)
1. Make a free account at github.com (if you don't have one).
2. Install **GitHub Desktop** (desktop.github.com) and sign in.
3. File -> **Add local repository** -> choose the `exec-dashboard` folder ->
   it says "not a git repository" -> click **create a repository** -> Create repository.
4. **Check the list of files on the left: `.env` must NOT be there.** (`.gitignore` blocks it,
   together with `.venv`, `data` and `logs`.) If you see `.env`, stop and tell Claude.
5. Click **Publish repository**. Keep **"Keep this code private" TICKED**.

## 1b. The daily schedule file (done)
The schedule lives in `.github/workflows/daily.yml` on GitHub (daily run plus a backup run that is
skipped if the day already succeeded). Claude can't write into `.github` folders (a safety rule), so
edit it on github.com, then **Fetch origin / Pull** in GitHub Desktop.

## 2. Give GitHub the two secrets (encrypted, nobody can read them back)
On github.com -> your repo -> **Settings** -> **Secrets and variables** -> **Actions** ->
**New repository secret**, twice:
| Name | Value |
|---|---|
| `DATABASE_URL` | everything after `DATABASE_URL=` in your `.env` |
| `GEMINI_API_KEY` | your new Gemini key |
| `FINNHUB_API_KEY` | your Finnhub key (news summaries; skipped if missing) |

## 3. Test it once by hand
Repo -> **Actions** tab -> **daily-pipeline** -> **Run workflow**. Wait ~5 minutes.
All steps green = done. From now on it runs by itself every morning (~06:15 Cairo).

## 4. Alarms (already on by default)
- GitHub emails you if a run fails (github.com -> Settings -> Notifications -> Actions).
- Supabase emails you about a week before it would pause a project.
- If it ever pauses anyway: Supabase dashboard -> the project -> **Resume project**
  (data is kept for up to a year).

## Before the defense (backup, one week before)
- Freeze the study brief: `python -m dashboard.ai --freeze`.
- Save a database snapshot to your laptop (ask Claude for the one-line command).
- Record a short screen video of the dashboard.
