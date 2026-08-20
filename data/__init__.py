"""
Campaign data: what a calculation is, where its results go, and how they
come back off disk.

  data.schema   -- StatePoint, CPDFTLabel, MDResult, CommitteeStats.
                   Plain dataclasses, JSON round-trippable, no I/O.
  data.store    -- append-only JSONL stores, safe for concurrent writers
                   inside one allocation.
  data.harvest  -- read finished run directories -> records. The ONLY place
                   that decides whether a calculation succeeded.
  data.xyz      -- export labels to CP-MACE extended-XYZ
                   (electron=/potential= tags).
"""

from data.schema import (
    StatePoint, CPDFTLabel, MDResult, CommitteeStats, RunStatus,
    state_point_id,
)
from data.store import JSONLStore, LabelStore, RecordStore
from data.harvest import harvest_jdftx_run, harvest_md_run, harvest_all
from data.xyz import write_cp_mace_xyz, append_cp_mace_frame

__all__ = [
    "StatePoint", "CPDFTLabel", "MDResult", "CommitteeStats", "RunStatus",
    "state_point_id",
    "JSONLStore", "LabelStore", "RecordStore",
    "harvest_jdftx_run", "harvest_md_run", "harvest_all",
    "write_cp_mace_xyz", "append_cp_mace_frame",
]
