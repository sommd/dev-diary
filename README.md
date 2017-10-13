# Developer Diary

A small (one file) command line tool for keeping a developer diary. Made for UNSW COMP2041 assignments, but could be useful for other things too.

## Installation

### Command

```sh
git clone https://github.com/sommd/dev-diary.git &&
pip install ./dev-diary
```

### Single File

```sh
curl -o diary.py 'https://raw.githubusercontent.com/sommd/dev-diary/master/diary.py' &&
chmod +x diary.py &&
pip install click sqlalchemy python-dateutil
```

This is useful when you just need it for a single project and want to share in across repos.

## Usage

```
Usage: diary [OPTIONS] COMMAND [ARGS]...

  Keep track of development time and tasks completed.

  Diary is kept track of using an SQLite database, 'diary.db' by default.

Options:
  --database PATH  The SQLite database file to use.
  --debug
  -q, --quiet
  --commit         Git commit the database afterwards.
  --help           Show this message and exit.

Commands:
  cancel  Stop the current session without recording it.
  edit    Edit a previous entry.
  entry   Add an entry to the diary, ignoring the current session.
  merge   Add all the entries of another diary into this one.
  show    Show entries in the diary.
  start   Start a new session with the current time.
  status  Show the status of the current session, if there is one.
  stop    Stop and record the current session with the current time.
  update  Download the latest version of diary.py and update.
```

### Example

```sh
$ diary status
No current session
$ diary start
Started new session at 2017-10-07 16:49:41.247897.
$ diary stop "Made some changes"
Recorded session: Date=2017-10-07, Start=16:49, Stop=16:50, Activity=Coding, Comments=Made some changes.
$ diary show
| Date       | Start | Stop  | Activity | Comments          |
| ---------- | ----- | ----- | -------- | ----------------- |
| 2017-10-07 | 16:49 | 16:50 | Coding   | Made some changes |
```
