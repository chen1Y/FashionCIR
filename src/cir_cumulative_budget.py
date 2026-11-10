"""Persistent cumulative API allocations; integer micro-CNY and no budget reset."""
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import sqlite3


def micro(value):
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError('Invalid money amount')
    return int((value*1000000).to_integral_value(rounding=ROUND_CEILING))


class CumulativeBudget:
    def __init__(self,path,cap_cny=300):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY,cap INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY,upper INTEGER NOT NULL,evidence TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,cap INTEGER NOT NULL,charged INTEGER,
          state TEXT NOT NULL,protocol_sha256 TEXT NOT NULL);
        ''')
        old=self.db.execute('SELECT cap FROM config WHERE id=1').fetchone()
        if old and old[0]!=micro(cap_cny):
            self.db.close()
            raise ValueError('Cumulative ceiling cannot be reset or changed')
        self.db.execute('INSERT OR IGNORE INTO config VALUES(1,?)',(micro(cap_cny),));self.db.commit()

    def import_history(self,entry_id,upper_cny,evidence):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old=self.db.execute('SELECT upper FROM history WHERE id=?',(entry_id,)).fetchone()
            amount=max(old[0] if old else 0,micro(upper_cny))
            self.db.execute('INSERT OR REPLACE INTO history VALUES(?,?,?)',(entry_id,amount,json.dumps(evidence,sort_keys=True)))
            if self._used()>self._cap():
                raise ValueError('Historical evidence exceeds cumulative ceiling')

    def _cap(self):
        return self.db.execute('SELECT cap FROM config WHERE id=1').fetchone()[0]

    def _used(self):
        history=self.db.execute('SELECT COALESCE(SUM(upper),0) FROM history').fetchone()[0]
        batches=self.db.execute('SELECT COALESCE(SUM(CASE WHEN charged IS NULL THEN cap ELSE charged END),0) FROM batches').fetchone()[0]
        return history+batches

    def reserve(self,batch_id,cap_cny,protocol_sha256):
        amount=micro(cap_cny)
        if not 0<amount<=5000000:
            raise ValueError('Initial small batches must reserve at most5 CNY')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.db.execute('SELECT 1 FROM batches WHERE id=?',(batch_id,)).fetchone():
                raise ValueError('Existing batch allocation cannot be reset or blindly repeated')
            if self._used()+amount>self._cap():
                raise ValueError('Cumulative API ceiling reached')
            self.db.execute("INSERT INTO batches VALUES(?,?,NULL,'reserved',?)",(batch_id,amount,protocol_sha256))

    def settle(self,batch_id,upper_cny,no_inflight):
        if not no_inflight:
            raise ValueError('Keep full batch reservation while requests may be in flight')
        amount=micro(upper_cny)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT cap,state FROM batches WHERE id=?',(batch_id,)).fetchone()
            if row is None or row[1]!='reserved' or amount>row[0]:
                raise ValueError('Invalid settlement; retain allocation for inspection')
            self.db.execute("UPDATE batches SET charged=?,state='settled' WHERE id=?",(amount,batch_id))

    def summary(self):
        return dict(cap_cny=self._cap()/1e6,accounted_upper_cny=self._used()/1e6,
            history=[dict(id=i,upper_cny=v/1e6,evidence=json.loads(e)) for i,v,e in self.db.execute('SELECT id,upper,evidence FROM history ORDER BY id')],
            batches=[dict(id=i,cap_cny=c/1e6,upper_cny=None if v is None else v/1e6,state=s,protocol_sha256=h)
                     for i,c,v,s,h in self.db.execute('SELECT * FROM batches ORDER BY id')],
            note='Cumulative project allocation, including historical uncertainty reserves; not an invoice or account balance')
