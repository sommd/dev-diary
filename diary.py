#!/usr/bin/env python3

import click
import inspect
from sqlalchemy import create_engine, Column, Integer, DateTime, String, Enum as SqlEnum, UniqueConstraint, CheckConstraint
from sqlalchemy.orm.session import Session
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from abc import ABC, abstractmethod


def auto_repr(*attrs):
    def __repr__(self):
        return "{}({})".format(
            type(self).__qualname__,
            ", ".join(
                "{}={}".format(attr, repr(getattr(self, attr))) for attr in attrs
            )
        )
    return __repr__


class Diary:
    Base = declarative_base()

    class Entry(Base):
        class Activity(Enum):
            CODING = "Coding"
            DEBUGGING = "Debugging"

        __tablename__ = "entries"

        id = Column(Integer, primary_key=True)
        start = Column(DateTime, nullable=False)
        stop = Column(DateTime, nullable=False, default=datetime.now)
        activity = Column(SqlEnum(Activity), nullable=False, default=Activity.CODING)
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


@diary.command()
@with_diary
def show(diary: Diary):
    if diary.entries.count():
        for entry in diary.entries:
            click.echo(entry)
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
