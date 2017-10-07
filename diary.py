#!/usr/bin/env python3

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from dateutil.parser import parse as parse_datetime
from functools import wraps
from sqlalchemy import create_engine, func, Column, Integer, DateTime, String, Enum, UniqueConstraint, CheckConstraint
from sqlalchemy.engine import Connectable
from sqlalchemy.pool import NullPool
from sqlalchemy.orm.session import Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Iterable, Sequence, Mapping, Callable, Any
import click
import enum


# Utils

def auto_repr(*attrs):
    """Automatic implementation of `__repr__` from a list of attributes."""

    def __repr__(self):
        return "{}({})".format(
            type(self).__qualname__,
            ", ".join(
                "{}={}".format(attr, repr(getattr(self, attr))) for attr in attrs
            )
        )
    return __repr__


# https://stackoverflow.com/a/14620633
class AttrDict(dict):
    """Extension of dict that `set`/`get`/`del` as attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class DateTimeParamType(click.ParamType):
    """Click `ParamType` for `datetime` objects. Automatically parses argument to `datetime` using `dateutil`."""

    name = "datetime"

    def convert(self, value, param, ctx):
        if value is None or isinstance(value, datetime):
            return value
        try:
            return parse_datetime(value)
        except ValueError as e:
            self.fail("Unknown date format '{}'".format(value))


DATE_TIME = DateTimeParamType()


class MappedParamType(click.ParamType):
    """Click `ParamType` for `Mapping` objects. Converts to value associated with the key entered."""

    def __init__(self, mapping: Mapping[str, Any]):
        self.mapping = mapping

    def get_metavar(self, param):
        return "[{}]".format("|".join(self.mapping))

    def convert(self, value, param, ctx):
        # Return current value if not a str, i.e. default value
        if not isinstance(value, str):
            return value

        try:
            return self.mapping[value]
        except KeyError:
            self.fail("Must be one of {}.".format(", ".join(
                "'{}'".format(key) for key in self.mapping.keys()
            )))


# Model

class Diary:
    Base = declarative_base()

    class Entry(Base):
        """Table of all previous entries in diary."""

        class Activity(enum.Enum):
            CODING = "Coding"
            DEBUGGING = "Debugging"

        __tablename__ = "entries"

        id = Column(Integer, primary_key=True)
        start = Column(DateTime, nullable=False)
        stop = Column(DateTime, nullable=False, default=datetime.now)
        activity = Column(Enum(Activity), nullable=False, default=Activity.CODING)
        comments = Column(String, nullable=False)

        __repr__ = auto_repr("id", "start", "stop", "activity", "comments")

    class Current(Base):
        """Table containing singleton with data on the current session, if any."""

        __tablename__ = "current"
        __table_args__ = (
            UniqueConstraint("id"),
            CheckConstraint("id=1")
        )

        id = Column(Integer, primary_key=True, default=1)
        start = Column(DateTime, nullable=False, default=datetime.now)

        __repr__ = auto_repr("id", "start")

    class Error(Exception, enum.Enum):
        NO_SESSION = "No current session"
        ALREADY_STARTED = "Session already started"

        def __str__(self):
            return str(self.value)

    @classmethod
    def from_path(cls, path: str, init=True, *args, **kwargs):
        # Create engine with no connection pooling so it automatically closes
        engine = create_engine(path, poolclass=NullPool, *args, **kwargs)

        if init:
            cls.init(engine)

        session = Session(bind=engine)
        return cls(session)

    @classmethod
    def init(cls, bind: Connectable):
        """Create all tables for a diary in the given `bind`."""
        cls.Base.metadata.create_all(bind)

    def __init__(self, session: Session):
        self._session = session

    def close(self):
        self._session.close()

    @contextmanager
    def session(self, commit=True):
        """Helper for automatically committing/rolling back using a `with` statement."""
        try:
            yield self._session
            if commit:
                self._session.commit()
        except:
            self._session.rollback()
            raise

    @property
    def entries(self) -> Entry:
        with self.session(False) as session:
            return session.query(self.Entry)

    @property
    def current(self) -> Current:
        with self.session(False) as session:
            return session.query(self.Current).one()

    @property
    def has_current(self) -> bool:
        with self.session(False) as session:
            return session.query(self.Current).count() != 0

    def start(self, time: datetime = None):
        """Start a new session with `time` as the start time, or the current time if `None`."""
        if self.has_current:
            raise self.Error.ALREADY_STARTED

        with self.session() as session:
            current = self.Current(start=time)
            session.add(current)
            return current

    def stop(self, comments: str, time: datetime = None, activity: Entry.Activity = None):
        """End the current session and record it into `entries`."""
        if not self.has_current:
            raise self.Error.NO_SESSION

        with self.session() as session:
            current = self.current
            entry = self.add_entry(current.start, comments, time, activity)
            session.delete(current)
            return entry

    def cancel(self):
        """End the current session without recording it into `entries`."""
        if not self.has_current:
            raise self.Error.NO_SESSION

        with self.session() as session:
            session.delete(self.current)

    def add_entry(self, start: datetime, comments: str, stop: datetime = None, activity: Entry.Activity = None):
        """Explicitly add an entry to `entries`. Works independently of the current session."""
        with self.session() as session:
            entry = self.Entry(start=start, stop=stop, activity=activity, comments=comments)
            session.add(entry)
            return entry


# Formatters

class EntryFormatter(ABC):
    """ABC interface for formating an `Entry` into a `str` to be output."""

    _default_fields = {
        "Date": lambda e: e.start.strftime("%Y-%m-%d"),
        "Start": lambda e: e.start.strftime("%H:%M"),
        "Stop": lambda e: e.stop.strftime("%H:%M"),
        "Activity": lambda e: e.activity.value,
        "Comments": lambda e: e.comments
    }

    def __init__(self, header=True, trailer=True, fields: Mapping[str, Callable[[Diary.Entry], str]] = _default_fields):
        self.use_header = header
        self.use_trailer = trailer
        self.fields = fields

    def format(self, entries: Iterable[Diary.Entry]) -> str:
        """Output a str representation of each entry given."""
        return "{}{}{}".format(
            self.header if self.use_header else "",
            self.separator.join(self.format_entry(entry) for entry in entries),
            self.trailer if self.use_trailer else ""
        )

    @property
    def header(self) -> str:
        """Header to be prepended to output of `format`, if enabled."""
        return ""

    @property
    def separator(self) -> str:
        """Separator between each entry, always enabled."""
        return "\n"

    @property
    def trailer(self) -> str:
        """Trailer to be appended to output of `format`, if enabled."""
        return ""

    @abstractmethod
    def format_entry(self, entry: Diary.Entry) -> str:
        """Str representation of a single entry."""
        raise NotImplementedError()


class BasicEntryFormatter(EntryFormatter):
    """Basic `EntryFormatter` which prints comma separated, 'field=value' pairs."""

    def format_entry(self, entry: Diary.Entry) -> str:
        return ", ".join("{}={}".format(name, field(entry)) for name, field in self.fields.items())


class MarkdownEntryFormatter(EntryFormatter):
    """`EntryFormatter` implementation which outputs as a Markdown table."""

    def __init__(self, *args, pretty=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretty = pretty

    def format(self, entries, header=True, trailer=True):
        widths = self._calculate_field_widths(entries) if self.pretty else None

        # Header
        string = self.get_header(widths) if header else ""
        # Entries
        string += self.separator.join(self.format_entry(entry, widths) for entry in entries)
        # Trailer
        if trailer:
            string += self.trailer

        return string

    def _calculate_field_widths(self, entries):
        widths = [len(field) for field in self.fields]

        # Calculate max width of each item
        for entry in entries:
            for i, field in enumerate(self.fields.values()):
                widths[i] = max(widths[i], len(field(entry)))

        return widths

    def get_header(self, widths: Sequence = None) -> str:
        if widths is not None:
            # Left justify if given widths
            titles = (field.ljust(width) for field, width in zip(self.fields, widths))
            dividers = ("-" * width for width in widths)
        else:
            titles = self.fields
            dividers = ("---" for i in range(len(self.fields)))

        fmt = "| {} |\n| {} |\n" if self.pretty else "{}\n{}\n"
        return fmt.format(" | ".join(titles), " | ".join(dividers))

    header = property(get_header)

    def format_entry(self, entry: Diary.Entry, widths: Sequence = None) -> str:
        fields = (field(entry) for field in self.fields.values())

        if widths is not None:
            # Left justify if given widths
            fields = (field.ljust(width) for field, width in zip(fields, widths))

        string = " | ".join(fields)
        return "| {} |".format(string) if self.pretty else string


# Command setup

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--database", type=click.Path(dir_okay=False), default="diary.db",
              help="The SQLite database file to use.")
@click.option("--debug", default=False, is_flag=True)
@click.option("-q", "--quiet", default=False, is_flag=True)
def diary(ctx, database: str, debug: bool, quiet: bool):
    """Keep track of development time and tasks completed.

    Diary is kept track of using an SQLite database, 'diary.db' by default.
    """

    # Create g and populate with shared arguments
    g = ctx.obj = AttrDict()
    g.debug = debug
    g.quiet = quiet

    # Open diary
    g.diary = Diary.from_path("sqlite:///{}".format(database), echo=g.debug)

    if not g.quiet and ctx.invoked_subcommand is None:
        ctx.invoke(show)
        click.echo()
        ctx.invoke(status)


def diary_command(*args, **kwargs):
    """Decorator for creating a subcommand of `diary` and automatically injecting `g` object as first paramater."""
    def decorate(fn):
        @diary.command(*args, **kwargs)
        @click.pass_context
        @wraps(fn)
        def decorator(ctx, *args, **kwargs):
            try:
                fn(ctx.obj, *args, **kwargs)
            except Diary.Error as e:
                raise click.UsageError(e)
        return decorator
    return decorate


@diary.resultcallback()
@click.pass_context
def close_database(ctx, result, *args, **kwargs):
    """Close connection to database when done."""
    ctx.obj.diary.close()


# Custom Types

class EntryParamType(click.types.IntParamType):
    name = "id"

    def get_metavar(self, param):
        return "[last|recent|INTEGER]"

    def convert(self, value, param, ctx):
        query = ctx.obj.diary.entries

        if value.lower() == "last":
            query = query.order_by(Diary.Entry.id.desc())
        elif value.lower() == "recent":
            query = query.order_by(Diary.Entry.stop.desc())
        else:
            id = super().convert(value, param, ctx)
            query = query.filter(Diary.Entry.id == id)

        return query.first()


ENTRY = EntryParamType()

FORMATTER = MappedParamType({
    "basic": BasicEntryFormatter(),
    "markdown": MarkdownEntryFormatter(),
    "markdown-basic": MarkdownEntryFormatter(pretty=False)
})

ACTIVITY = MappedParamType({
    activity.name.lower(): activity for activity in Diary.Entry.Activity
})


# Commands

@diary_command()
@click.option("-f", "--format", type=FORMATTER, default="markdown")
def show(g, format: EntryFormatter):
    """Show entries in the diary."""
    if g.diary.entries.count():
        click.echo(format.format(g.diary.entries.order_by(Diary.Entry.start)))
    else:
        click.echo("No entries")


@diary_command()
def status(g):
    """Show the status of the current session, if there is one."""
    if g.diary.has_current:
        start = g.diary.current.start
        duration = datetime.now() - start
        start = start.strftime("%H:%M")
        duration = int(duration.total_seconds() / 60)
        click.echo("Started at {} ({} minutes ago)".format(start, duration))
    else:
        click.echo("No current session")


@diary_command()
@click.argument("time", type=DATE_TIME, default=datetime.now())
@click.option("-f", "--force", default=False, is_flag=True, help="Start a new session even if one is already active.")
def start(g, time: datetime, force: bool):
    """Start a new session with the current time."""
    if force and g.diary.has_current:
        g.diary.cancel()

    current = g.diary.start(time)

    if not g.quiet:
        click.echo("Started new session at {}.".format(current.start.isoformat(" ")))


@diary_command()
@click.argument("comments")
@click.argument("time", type=DATE_TIME, default=datetime.now())
@click.option("-a", "--activity", type=ACTIVITY, default="coding")
def stop(g, comments: str, time: datetime, activity: Diary.Entry.Activity):
    """Stop and record the current session with the current time."""
    entry = g.diary.stop(comments, activity=activity, time=time)

    if not g.quiet:
        _echo_entry(entry, "Recorded session: {}.")


def _echo_entry(entry: Diary.Entry, fmt: str = "{}"):
    # Lazily create EntryFormatter
    if not hasattr(_echo_entry, "formatter"):
        _echo_entry.formatter = BasicEntryFormatter()
    formatter = _echo_entry.formatter

    click.echo(fmt.format(formatter.format_entry(entry)))


@diary_command()
@click.confirmation_option(prompt="Are you sure you want to cancel the current session?")
def cancel(g):
    """Stop the current session without recording it."""
    g.diary.cancel()

    if not g.quiet:
        click.echo("Current session cancelled.")


@diary_command()
@click.argument("start", type=DATE_TIME)
@click.argument("comments")
@click.option("-s", "--stop", type=DATE_TIME, default=datetime.now())
@click.option("-a", "--activity", type=ACTIVITY, default="coding")
def entry(g, start: datetime, comments: str, stop: datetime, activity: Diary.Entry.Activity):
    """Add an entry to the diary, ignoring the current session."""
    entry = g.diary.add_entry(start, comments, stop, activity)

    if not g.quiet:
        _echo_entry(entry, "Added entry: {}.")


@diary_command()
@click.argument("other", type=click.Path(exists=True, dir_okay=False))
def merge(g, other: str):
    """Add all the entries of another diary into this one."""
    other = Diary.from_path("sqlite:///{}".format(other), False, echo=g.debug)

    count = 0
    for entry in other.entries:
        g.diary.add_entry(entry.start, entry.comments, entry.stop, entry.activity)

        if not g.quiet:
            _echo_entry(entry, "Added entry: {}.")
            count += 1

    if not g.quiet:
        click.echo("Added {} entries.".format(count))


@diary_command()
@click.argument("entry", type=ENTRY, default="last")
@click.option("-c", "--comments")
@click.option("-a", "--activity", type=ACTIVITY)
@click.option("--start", type=DATE_TIME)
@click.option("--stop", type=DATE_TIME)
def edit(g, entry: Diary.Entry, comments: str, activity: Diary.Entry.Activity, start: datetime, stop: datetime):
    """Edit a previous entry.

    ENTRY can be an id, 'last' for the last added entry, or 'recent' for the most recent entry (by stop time).
    """

    with g.diary.session():
        if comments is not None:
            entry.comments = comments
        if activity is not None:
            entry.activity = activity
        if start is not None:
            entry.start = start
        if stop is not None:
            entry.stop = stop

    _echo_entry(entry, "Edited entry: {}.")


if __name__ == "__main__":
    diary()
