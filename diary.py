#!/usr/bin/env python3

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from sqlalchemy import create_engine, Column, Integer, DateTime, String, Enum, UniqueConstraint, CheckConstraint
from sqlalchemy.orm.session import Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Iterable, Sequence, Mapping, Callable
import click
import enum


# Utils

def auto_repr(*attrs):
    def __repr__(self):
        return "{}({})".format(
            type(self).__qualname__,
            ", ".join(
                "{}={}".format(attr, repr(getattr(self, attr))) for attr in attrs
            )
        )
    return __repr__


# Model

class Diary:
    Base = declarative_base()

    class Entry(Base):
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
        __tablename__ = "current"
        __table_args__ = (
            UniqueConstraint("id"),
            CheckConstraint("id=1")
        )

        id = Column(Integer, primary_key=True, default=1)
        start = Column(DateTime, nullable=False, default=datetime.now)

        __repr__ = auto_repr("id", "start")

    def __init__(self, session: Session):
        self._session = session

    @contextmanager
    def session(self, commit=True):
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
            return session.query(self.Entry).order_by(self.Entry.start)

    @property
    def current(self) -> Current:
        with self.session(False) as session:
            return session.query(self.Current).one()

    @property
    def has_current(self) -> bool:
        with self.session(False) as session:
            return session.query(self.Current).count() != 0

    def start(self, time: datetime = None):
        if self.has_current:
            raise ValueError("Session already started")

        with self.session() as session:
            current = self.Current(start=time)
            session.add(current)
            return current

    def stop(self, comments: str, time: datetime = None, activity: Entry.Activity = None):
        if not self.has_current:
            raise ValueError("No current session")

        with self.session() as session:
            current = self.current
            entry = self.add_entry(current.start, comments, time, activity)
            session.delete(current)
            return entry

    def cancel(self):
        if not self.has_current:
            raise ValueError("No current session")

        with self.session() as session:
            session.delete(self.current)

    def add_entry(self, start: datetime, comments: str, stop: datetime = None, activity: Entry.Activity = None):
        with self.session() as session:
            entry = self.Entry(start=start, stop=stop, activity=activity, comments=comments)
            session.add(entry)
            return entry


# Formatters

class EntryFormatter(ABC):
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
        return "{}{}{}".format(
            self.header if self.use_header else "",
            self.separator.join(self.format_entry(entry) for entry in entries),
            self.trailer if self.use_trailer else ""
        )

    @property
    def header(self) -> str:
        return ""

    @property
    def separator(self) -> str:
        return "\n"

    @property
    def trailer(self) -> str:
        return ""

    @abstractmethod
    def format_entry(self, entry: Diary.Entry) -> str:
        raise NotImplementedError()


class BasicEntryFormatter(EntryFormatter):
    def format_entry(self, entry: Diary.Entry) -> str:
        return ", ".join("{}={}".format(name, field(entry)) for name, field in self.fields.items())


class MarkdownEntryFormatter(EntryFormatter):
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


# Commands

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--database", type=click.Path(dir_okay=False), default="diary.db")
@click.option("--debug", default=False, is_flag=True)
def diary(ctx, database: str, debug: bool):
    # Connect to DB
    engine = create_engine("sqlite:///{}".format(database), echo=debug)
    # Init DB
    Diary.Base.metadata.create_all(engine)
    # Create model
    session = Session(bind=engine)
    diary = Diary(session)

    # Add to ctx
    ctx.obj.update(
        engine=engine,
        session=session,
        diary=diary
    )

    if ctx.invoked_subcommand is None:
        ctx.invoke(show)
        click.echo()
        ctx.invoke(status)


def with_diary(fn):
    @click.pass_context
    @wraps(fn)
    def decorator(ctx, *args, **kwargs):
        fn(ctx.obj["diary"], *args, **kwargs)
    return decorator


@diary.resultcallback()
@click.pass_context
def close_database(ctx, result, *args, **kwargs):
    ctx.obj["session"].close()
    ctx.obj["engine"].dispose()


formatters = {
    "basic": BasicEntryFormatter(),
    "markdown": MarkdownEntryFormatter(),
    "markdown-basic": MarkdownEntryFormatter(pretty=False)
}


@diary.command()
@click.option("-f", "--format", type=click.Choice(formatters), default="markdown")
@with_diary
def show(diary: Diary, format: str):
    formatter = formatters[format]

    if diary.entries.count():
        click.echo(formatter.format(diary.entries))
    else:
        click.echo("No entries")


@diary.command()
@with_diary
def status(diary: Diary):
    if diary.has_current:
        start = diary.current.start
        duration = datetime.now() - start
        start = start.strftime("%H:%M")
        duration = int(duration.total_seconds() / 60)
        click.echo("Started at {} ({} minutes ago)".format(start, duration))
    else:
        click.echo("No current session")


@diary.command()
@with_diary
def start(diary: Diary):
    diary.start()


@diary.command()
@click.argument("comments")
@with_diary
def stop(diary: Diary, comments: str):
    diary.stop(comments)


@diary.command()
@with_diary
def cancel(diary: Diary):
    diary.cancel()


@diary.command()
@with_diary
def entry(diary: Diary):
    # TODO
    click.echo("TODO")


if __name__ == "__main__":
    diary(obj={})
